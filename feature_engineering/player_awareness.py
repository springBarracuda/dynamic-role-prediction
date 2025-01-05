import numpy as np

class PlayerAwarenessCalculator:
    def __init__(self, beta=0.04, alpha=0.1, field_of_view=180):
        """
        Initializes the Player Awareness Calculator.

        Args:
            beta (float): Distance decay factor.
            alpha (float): Speed scaling factor.
            field_of_view (float): Field of view in degrees (e.g., 90 means ±45° from orientation).
        """
        self.beta = beta
        self.alpha = alpha
        self.field_of_view = np.radians(field_of_view / 2)  # Convert FOV to radians and halve it

    def convert_angle(self, angle):
        """
        Convert an angle from the positive x-axis (counter-clockwise) to the positive y-axis (clockwise).

        Args:
            angle (float or np.ndarray): Angle(s) in radians to convert.

        Returns:
            float or np.ndarray: Converted angle(s) in radians.
        """
        # Subtract from π/2 to switch reference axis
        converted_angle = (np.pi / 2) - angle

        # Normalize to [0, 2π) range
        converted_angle = converted_angle % (2 * np.pi)

        return converted_angle

    def calculate_awareness(self, x, y, o, s, p_x, p_y):
        """
        Single awareness calculation for a player and a target point.

        Args:
            x, y (float): Player's position.
            o (float): Player's orientation in degrees.
            s (float): Player's speed.
            p_x, p_y (float): Target point's position.

        Returns:
            float: Awareness score for the target point.
        """
        dx, dy = p_x - x, p_y - y
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_target = self.convert_angle(np.arctan2(dy, dx))  # Angle to the target point
        player_orientation = np.radians(o)  # Convert orientation to radians
        angle_offset = np.abs(player_orientation - angle_to_target)  # Absolute angle offset
        angle_offset = np.minimum(angle_offset, 2 * np.pi - angle_offset)  # Normalize to [0, π]

        # Check if target is within the field of view
        if angle_offset > self.field_of_view:
            return 0

        # Compute awareness score
        awareness = (1 + self.alpha * s) * np.exp(-self.beta * distance) * np.cos(angle_offset)
        return max(0, awareness)  # Ensure no negative awareness

    def calculate_awareness_batch(self, positions, attributes, targets):
        """
        Batch awareness calculation for multiple players and multiple target points.

        Args:
            positions (np.ndarray): Array of shape (N, 2) containing player positions [x, y].
            attributes (np.ndarray): Array of shape (N, 2) containing player attributes [o, s].
            targets (np.ndarray): Array of shape (M, 2) containing target positions [p_x, p_y].

        Returns:
            np.ndarray: Array of shape (N, M) with awareness scores for each player-target pair.
        """
        x, y = positions[:, 0:1], positions[:, 1:2]  # Shape (N, 1)
        o, s = attributes[:, 0:1], attributes[:, 1:2]  # Shape (N, 1)
        p_x, p_y = targets[:, 0], targets[:, 1]  # Shape (M,)

        # Compute differences in x and y coordinates
        dx = p_x - x  # Shape (N, M)
        dy = p_y - y  # Shape (N, M)

        # Compute distances
        distance = np.sqrt(dx**2 + dy**2)  # Shape (N, M)

        # Compute angles to targets
        angle_to_target = self.convert_angle(np.arctan2(dy, dx))  # Shape (N, M)
        player_orientation = np.radians(o)  # Shape (N, 1) broadcast over (N, M)
        angle_offset = np.abs(player_orientation - angle_to_target)  # Shape (N, M)
        angle_offset = np.minimum(angle_offset, 2 * np.pi - angle_offset)  # Compute smaller offset

        # Mask targets outside the field of view
        in_fov_mask = angle_offset <= self.field_of_view  # Shape (N, M)

        # Compute awareness
        awareness = (
            (1 + self.alpha * s) * np.exp(-self.beta * distance) * np.cos(angle_offset)
        )  # Shape (N, M)

        # Apply field of view mask
        awareness[~in_fov_mask] = 0

        # Ensure non-negative values
        awareness = np.maximum(0, awareness)

        return awareness
