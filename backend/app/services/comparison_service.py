
import cv2
import numpy as np

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


class ComparisonService:

    @staticmethod
    def decode_image(data: bytes, content_type: str):
        if content_type not in ALLOWED_TYPES:
            raise ValueError(
                "Only JPEG, PNG and WebP images are supported."
            )

        if not data:
            raise ValueError("Uploaded image is empty.")

        if len(data) > MAX_FILE_SIZE:
            raise ValueError("Maximum file size is 10 MB.")

        raw = np.frombuffer(data, dtype=np.uint8)

        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid or corrupted image.")

        height, width = image.shape[:2]

        if height * width > MAX_PIXELS:
            raise ValueError(
                "Image exceeds the 20-megapixel limit."
            )

        return image

    @staticmethod
    def image_info(image):
        height, width = image.shape[:2]

        return {
            "width": width,
            "height": height,
            "channels": image.shape[2]
        }

    def validate_pair(
        self,
        before_data,
        before_type,
        after_data,
        after_type
    ):
        before = self.decode_image(
            before_data, before_type
        )

        after = self.decode_image(
            after_data, after_type
        )

        return {
            "status": "validated",
            "before": self.image_info(before),
            "after": self.image_info(after),
            "same_dimensions": before.shape == after.shape,
            "message": (
                "Both images are valid and ready for comparison."
            )
        }


comparison_service = ComparisonService()
