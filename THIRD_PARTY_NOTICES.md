# Third-Party Notices

This repository's source code is released under the MIT License in [`LICENSE`](LICENSE).

Some optional integrations and model assets are governed by separate license terms from their original authors.

## Optional Ultralytics / YOLO Support

- `ultralytics` is not part of the default install in `requirements.txt`.
- YOLO-based features are opt-in through [`requirements-yolo.txt`](requirements-yolo.txt).
- If you enable `detector.type: yolo` or `detector.type: omniparser`, review Ultralytics' current license terms before distributing or commercializing that setup.

## Model Assets

- Files under `models/` may have license terms or redistribution limits separate from this repository's MIT license.
- In particular, verify the provenance and redistribution rights of any `.pt` model weights before publishing, sublicensing, or using them in a commercial context.
- If you trained the weights yourself, keep a short note describing the training source code, dataset, and intended license so future users can understand what they are allowed to do.
