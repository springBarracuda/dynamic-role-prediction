import numpy as np


class PlayerInfluenceCalculator:
    def __init__(self, beta=0.075, alpha=0.1):
        self.beta = beta
        self.alpha = alpha

    def convert_angle(self, angle):
        """
        Convert an angle from the positive x-axis (counter-clockwise) to the positive y-axis (clockwise).

        Args:
            angle (float or np.ndarray): Angle(s) in degrees to convert.

        Returns:
            float or np.ndarray: Converted angle(s) in degrees.
        """
        # Subtract from 90 to switch reference axis
        converted_angle = (np.pi / 2) - angle

        # Normalize to [0, 360) range
        converted_angle = converted_angle % (2 * np.pi)

        return converted_angle

    def calculate_influence(self, x, y, dir, o, s, p_x, p_y):
        """
        Single influence calculation (kept for reference or specific use cases).
        """
        dx, dy = p_x - x, p_y - y
        distance = np.sqrt(dx**2 + dy**2)
        angle_dir = np.radians(dir) - self.convert_angle(np.arctan2(dy, dx))
        angle_o = np.radians(o) - self.convert_angle(np.arctan2(dy, dx))
        return (2 + 0.7 * np.cos(angle_dir) + 0.3 * np.cos(angle_o)) * (1 + self.alpha * s) * np.exp(-self.beta * distance)

    def calculate_influence_batch(self, positions, attributes, targets):
        """
        Batch influence calculation.

        Args:
            positions (np.ndarray): Array of shape (N, 2) containing player positions [x, y].
            attributes (np.ndarray): Array of shape (N, 3) containing player attributes [dir, o, s].
            targets (np.ndarray): Array of shape (M, 2) containing target positions [p_x, p_y].

        Returns:
            np.ndarray: Array of shape (N, M) with influence scores for each player-target pair.
        """
        x, y = positions[:, 0:1], positions[:, 1:2]  # Shape (N, 1)
        dir, o, s = attributes[:, 0:1], attributes[:, 1:2], attributes[:, 2:3]  # Shape (N, 1)
        p_x, p_y = targets[:, 0], targets[:, 1]  # Shape (M,)

        # Compute differences in x and y coordinates
        dx = p_x - x  # Shape (N, M)
        dy = p_y - y  # Shape (N, M)

        # Compute distance matrix
        distance = np.sqrt(dx**2 + dy**2)  # Shape (N, M)

        # Compute angles to targets
        angle_to_target = self.convert_angle(np.arctan2(dy, dx))  # Shape (N, M)
        angle_dir = np.radians(dir) - angle_to_target  # Shape (N, M)
        angle_o = np.radians(o) - angle_to_target  # Shape (N, M)

        # Compute influence using vectorized operations
        influence = (
            (2 + 0.7 * np.cos(angle_dir) + 0.3 * np.cos(angle_o))
            * (1 + self.alpha * s)  # Shape (N, 1) is broadcast over (N, M)
            * np.exp(-self.beta * distance)
        )  # Shape (N, M)

        return influence