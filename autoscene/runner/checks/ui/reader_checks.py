from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from autoscene.core.exceptions import VerificationError
from autoscene.core.models import BoundingBox
from autoscene.runner.step_specs import ReaderValueInRangeCheckArgs, RegionSpec
from autoscene.vision.interfaces import ReaderAdapter


class ReaderUIChecks:
    def __init__(
        self,
        screenshot_actions: object | None = None,
        readers: dict[str, ReaderAdapter] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.screenshot_actions = screenshot_actions
        self.readers = dict(readers or {})
        self._read_all_cache: dict[tuple[str, object, object], dict[str, Any]] = {}
        self.logger = (
            logging.getLogger(self.__class__.__name__)
            if logger is None
            else logger.getChild(self.__class__.__name__)
        )
        self.handlers = {
            "reader_value_in_range": self._handle_reader_value_in_range,
        }
        self.typed_handlers = {
            "reader_value_in_range": self._handle_reader_value_in_range_typed,
        }

    def _handle_reader_value_in_range(self, params: dict[str, Any]) -> bool:
        reader_name, reader = self._resolve_reader(params)
        image = self._resolve_image(params)
        query = params.get("query")
        region = self._parse_region(params.get("region"))
        result = self._read_result(
            reader_name=reader_name,
            reader=reader,
            image=image,
            query=query,
            region=region,
        )
        min_score = float(params.get("min_score", 0.0))
        self.logger.info(
            "reader_value_in_range reader=%s query=%s raw_value=%r score=%.3f min_score=%.3f source=%s label=%s",
            reader_name,
            query,
            getattr(result, "value", None),
            float(getattr(result, "score", 0.0)),
            min_score,
            getattr(result, "source", ""),
            getattr(result, "label", ""),
        )
        if float(result.score) < min_score:
            self.logger.info(
                "reader_value_in_range reader=%s query=%s result=failed reason=score_below_threshold",
                reader_name,
                query,
            )
            return False

        actual_value = self._coerce_numeric(result.value, field_name="reader result value")
        if "expected" in params and params.get("expected") is not None:
            expected_value = self._coerce_numeric(params["expected"], field_name="expected")
            tolerance = float(params.get("tolerance", 0.0))
            minimum = expected_value - tolerance
            maximum = expected_value + tolerance
        else:
            if params.get("min") is None or params.get("max") is None:
                raise VerificationError(
                    "reader_value_in_range requires either 'expected' or both 'min' and 'max'."
                )
            minimum = self._coerce_numeric(params["min"], field_name="min")
            maximum = self._coerce_numeric(params["max"], field_name="max")
        passed = minimum <= actual_value <= maximum
        self.logger.info(
            "reader_value_in_range reader=%s query=%s actual=%.3f range=[%.3f, %.3f] expected=%s tolerance=%s result=%s",
            reader_name,
            query,
            actual_value,
            minimum,
            maximum,
            params.get("expected"),
            params.get("tolerance"),
            "passed" if passed else "failed",
        )
        return passed

    def _handle_reader_value_in_range_typed(self, args: ReaderValueInRangeCheckArgs) -> bool:
        reader_name, reader = self._resolve_reader({"reader": args.reader})
        image = self._resolve_image({"image_path": args.image_path})
        query = args.query
        region = self._region_spec_to_bounding_box(args.region)
        result = self._read_result(
            reader_name=reader_name,
            reader=reader,
            image=image,
            query=query,
            region=region,
        )
        min_score = float(args.min_score)
        self.logger.info(
            "reader_value_in_range reader=%s query=%s raw_value=%r score=%.3f min_score=%.3f source=%s label=%s",
            reader_name,
            query,
            getattr(result, "value", None),
            float(getattr(result, "score", 0.0)),
            min_score,
            getattr(result, "source", ""),
            getattr(result, "label", ""),
        )
        if float(result.score) < min_score:
            self.logger.info(
                "reader_value_in_range reader=%s query=%s result=failed reason=score_below_threshold",
                reader_name,
                query,
            )
            return False

        actual_value = self._coerce_numeric(result.value, field_name="reader result value")
        if args.expected is not None:
            expected_value = self._coerce_numeric(args.expected, field_name="expected")
            tolerance = float(args.tolerance)
            minimum = expected_value - tolerance
            maximum = expected_value + tolerance
        else:
            if args.min is None or args.max is None:
                raise VerificationError(
                    "reader_value_in_range requires either 'expected' or both 'min' and 'max'."
                )
            minimum = self._coerce_numeric(args.min, field_name="min")
            maximum = self._coerce_numeric(args.max, field_name="max")
        passed = minimum <= actual_value <= maximum
        self.logger.info(
            "reader_value_in_range reader=%s query=%s actual=%.3f range=[%.3f, %.3f] expected=%s tolerance=%s result=%s",
            reader_name,
            query,
            actual_value,
            minimum,
            maximum,
            args.expected,
            args.tolerance,
            "passed" if passed else "failed",
        )
        return passed

    def _read_result(
        self,
        *,
        reader_name: str,
        reader: ReaderAdapter,
        image: Any,
        query: Any,
        region: BoundingBox | None,
    ) -> Any:
        read_all = getattr(reader, "read_all", None)
        cache_key = self._make_read_all_cache_key(reader_name, image, region)
        if query is not None and cache_key is not None and callable(read_all):
            cached_results = self._read_all_cache.get(cache_key)
            if cached_results is None:
                cached_results = {
                    str(name): value for name, value in dict(read_all(image, region=region)).items()
                }
                self._read_all_cache[cache_key] = cached_results
            result = cached_results.get(str(query))
            if result is not None:
                return result
        return reader.read(image, query=query, region=region)

    def _resolve_reader(self, params: dict[str, Any]) -> tuple[str, ReaderAdapter]:
        if not self.readers:
            raise VerificationError("No readers configured. Add 'readers' to the test case.")
        raw_reader = params.get("reader")
        if raw_reader is None:
            if len(self.readers) == 1:
                return next(iter(self.readers.items()))
            raise VerificationError(
                "Reader checks require 'reader' when multiple readers are configured."
            )
        reader_name = str(raw_reader)
        reader = self.readers.get(reader_name)
        if reader is None:
            available = ", ".join(sorted(self.readers))
            raise VerificationError(
                f"Unknown reader '{reader_name}'. Available: {available}"
            )
        return reader_name, reader

    def _resolve_image(self, params: dict[str, Any]) -> Any:
        raw_image_path = params.get("image_path")
        if raw_image_path is None:
            screenshot_actions = self._require_screenshot_actions()
            return screenshot_actions.screenshot()
        image_path = Path(str(raw_image_path))
        if not image_path.exists():
            raise VerificationError(f"Reader check image does not exist: {image_path}")
        self.logger.info("reader_value_in_range use_image_path=%s", image_path)
        return str(image_path)

    def _require_screenshot_actions(self):
        if self.screenshot_actions is None:
            raise VerificationError("Reader checks require screenshot actions.")
        return self.screenshot_actions

    @staticmethod
    def _make_read_all_cache_key(
        reader_name: str,
        image: Any,
        region: BoundingBox | None,
    ) -> tuple[str, object, object] | None:
        if not isinstance(image, (str, Path)):
            return None
        path = Path(image)
        try:
            stat = path.stat()
        except OSError:
            return None
        region_key = None if region is None else (
            int(region.x1),
            int(region.y1),
            int(region.x2),
            int(region.y2),
        )
        image_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        return (reader_name, image_key, region_key)

    @staticmethod
    def _parse_region(value: Any) -> BoundingBox | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise VerificationError("Reader check field 'region' must be a mapping.")
        return BoundingBox(
            x1=int(value["x1"]),
            y1=int(value["y1"]),
            x2=int(value["x2"]),
            y2=int(value["y2"]),
        )

    @staticmethod
    def _region_spec_to_bounding_box(value: RegionSpec | None) -> BoundingBox | None:
        if value is None:
            return None
        return BoundingBox(
            x1=int(value.x1),
            y1=int(value.y1),
            x2=int(value.x2),
            y2=int(value.y2),
        )

    @staticmethod
    def _coerce_numeric(value: Any, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise VerificationError(f"Field '{field_name}' must be numeric, got {value!r}.") from exc
