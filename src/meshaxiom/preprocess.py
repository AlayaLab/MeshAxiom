"""One shared, recorded image preparation for both geometry networks."""
from pathlib import Path
from PIL import Image, ImageOps
from .contracts import MeshAxiomError, file_binding

def prepare_image(source, destination, *, background="auto", size=512):
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise MeshAxiomError("Input image does not exist: " + str(source))
    try:
        with Image.open(source) as opened:
            if opened.width * opened.height > 32_000_000:
                raise MeshAxiomError("Input image exceeds 32 million pixels")
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise MeshAxiomError("Cannot decode input image") from exc
    has_alpha = image.getchannel("A").getextrema()[0] < 255
    method = "supplied_alpha" if has_alpha else "keep"
    if background == "alpha" and not has_alpha:
        raise MeshAxiomError("background=alpha needs an image with foreground transparency")
    if background == "remove" or background == "auto" and not has_alpha:
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise MeshAxiomError("Install the background extra for RGB image matting, provide transparent PNG, or select background=keep") from exc
        session = new_session("u2net", providers=["CPUExecutionProvider"])
        image = remove(image, session=session).convert("RGBA")
        method = "rembg_u2net_cpu"
    if background == "keep":
        image.putalpha(255)
        method = "keep"
    bounds = image.getchannel("A").getbbox()
    if not bounds:
        raise MeshAxiomError("Image has no visible foreground")
    # Preserve the supplied frame in keep mode. Segmented inputs are centered
    # with a 5% border, and this transform is recorded rather than hidden.
    if method != "keep":
        image = image.crop(bounds)
    target = int(size * .9) if method != "keep" else size
    image.thumbnail((target, target), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.alpha_composite(image, offset)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(destination)
    return {"input": file_binding(source), "prepared": file_binding(destination), "background_method": method,
            "foreground_bounds": list(bounds), "resized_foreground": list(image.size), "offset": list(offset),
            "resolution": [size, size], "background_rgb": [0, 0, 0], "shared_by_vertex_and_topology": True}

