
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.comparison_service import (
    comparison_service,
    MAX_FILE_SIZE,
)


from app.services.image_comparison_engine import (
    image_comparison_engine
)

from app.services.image_alignment import ImageAlignmentService


router = APIRouter()

VALID_ALIGNMENT_MODES = ImageAlignmentService.VALID_MODES


async def read_limited(file: UploadFile):
    data = await file.read(MAX_FILE_SIZE + 1)

    if len(data) > MAX_FILE_SIZE:
        raise ValueError("Maximum file size is 10 MB.")

    return data


@router.post("/validate")
async def validate_images(
    before: UploadFile = File(...),
    after: UploadFile = File(...)
):
    try:
        before_data = await read_limited(before)
        after_data = await read_limited(after)

        return comparison_service.validate_pair(
            before_data,
            before.content_type,
            after_data,
            after.content_type
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    finally:
        await before.close()
        await after.close()


@router.post("/compare")
async def compare_images(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    alignment_mode: str = Form("auto"),
):
    try:
        # Validate alignment mode early.
        mode = alignment_mode.lower().strip()
        if mode not in VALID_ALIGNMENT_MODES:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "INVALID_ALIGNMENT_MODE",
                    "message": (
                        f"Unknown alignment mode '{alignment_mode}'. "
                        f"Accepted values: {sorted(VALID_ALIGNMENT_MODES)}."
                    ),
                },
            )

        before_data = await read_limited(before)
        after_data = await read_limited(after)

        before_image = comparison_service.decode_image(
            before_data,
            before.content_type
        )

        after_image = comparison_service.decode_image(
            after_data,
            after.content_type
        )

        return image_comparison_engine.compare(
            before_image,
            after_image,
            mode=mode,
        )

    except ValueError as error:
        error_msg = str(error)

        # Alignment failures get a structured error body.
        if "alignment failed" in error_msg.lower():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "ALIGNMENT_FAILED",
                    "message": error_msg,
                    "attempted_methods": (
                        ["SIFT", "AKAZE", "ORB"]
                        if mode == "auto"
                        else [mode.upper()]
                    ),
                    "failure_reasons": [
                        "Insufficient matching features between the two images."
                    ],
                    "guidance": (
                        "Ensure both images show the same scene from a "
                        "similar viewpoint with visible shared details.  "
                        "If the images are already pixel-aligned, you may "
                        "use alignment_mode='direct'."
                    ),
                },
            )

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    finally:
        await before.close()
        await after.close()
