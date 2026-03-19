from __future__ import annotations

from pathlib import Path
from typing import Any

from autoscene.core.exceptions import DependencyMissingError
from autoscene.core.models import BoundingBox

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None


class OpenCVAdapterBase:
    def _path_image_cache(self) -> dict[tuple[str, int, int], Any]:
        cache = getattr(self, "_path_image_cache_store", None)
        if cache is None:
            cache = {}
            setattr(self, "_path_image_cache_store", cache)
        return cache

    def require_cv2(self):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise DependencyMissingError(
                "opencv-python is not installed. Run: pip install opencv-python"
            ) from exc
        return cv2

    @staticmethod
    def require_numpy():
        if np is None:
            raise DependencyMissingError("numpy is not installed. Run: pip install numpy")
        return np

    @staticmethod
    def require_pillow():
        if Image is None:
            raise DependencyMissingError("Pillow is not installed. Run: pip install Pillow")
        return Image

    def to_ndarray(self, image: Any) -> Any:
        cv2 = self.require_cv2()
        numpy = self.require_numpy()

        if isinstance(image, Path):
            image = str(image)
        if isinstance(image, str):
            image_path = Path(image)
            try:
                stat = image_path.stat()
            except OSError as exc:
                raise FileNotFoundError(f"Unable to read image path: {image}") from exc
            cache_key = (str(image_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
            cache = self._path_image_cache()
            cached = cache.get(cache_key)
            if cached is not None:
                return cached.copy()
            loaded = cv2.imread(str(image_path))
            if loaded is None:
                raise FileNotFoundError(f"Unable to read image path: {image}")
            cache[cache_key] = loaded
            return loaded.copy()
        if numpy is not None and isinstance(image, numpy.ndarray):
            return image.copy()
        if Image is not None and isinstance(image, Image.Image):
            rgb_image = numpy.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        raise TypeError("Expected an image path, PIL image, or numpy.ndarray.")

    def clip_region(
        self,
        image: Any,
        region: BoundingBox | None = None,
    ) -> tuple[Any, tuple[int, int]]:
        frame = self.to_ndarray(image)
        if region is None:
            return frame, (0, 0)

        height, width = frame.shape[:2]
        x1 = max(0, min(width, int(region.x1)))
        y1 = max(0, min(height, int(region.y1)))
        x2 = max(0, min(width, int(region.x2)))
        y2 = max(0, min(height, int(region.y2)))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid region for OpenCV adapter.")
        return frame[y1:y2, x1:x2].copy(), (x1, y1)

    def preprocess(
        self,
        image: Any,
        *,
        grayscale: bool = False,
        blur_kernel: int | None = None,
    ) -> Any:
        cv2 = self.require_cv2()
        frame = self.to_ndarray(image)
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if blur_kernel and blur_kernel > 1:
            frame = cv2.GaussianBlur(frame, (int(blur_kernel), int(blur_kernel)), 0)
        return frame
