"""Request images: base64 data URLs decoded to RGB images for Gemma 4's vision tower.

The decision API places a request's images before its state. Every branch of the request shares
them, so their key-value blocks are computed once and reused by every option order. Remote URLs are
never fetched: a service that downloads on request is a server-side request forgery risk.
"""
import base64
import binascii
import io

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 64_000_000  # decompression-bomb guard; Gemma's processor resizes to the soft-token budget anyway
FORMATS = {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP', 'image/gif': 'GIF'}
# Gemma 4's per-image budget (`max_soft_tokens`) in the evaluated run; the checkpoint's own processor uses 280.
SOFT_TOKENS = 1120


def image_tokens(soft_tokens=SOFT_TOKENS):
    """Prompt tokens one `<|image|>` placeholder becomes inside TensorRT-LLM: BOI + soft tokens + EOI."""
    return soft_tokens + 2




def check(urls):
    """Shape checks that need no decoding (schema validation)."""
    if len(urls) > MAX_IMAGES:
        raise ValueError(f'At most {MAX_IMAGES} images per request')
    for i, url in enumerate(urls):
        head, sep, _ = url.partition(',')
        mime = head[5:].removesuffix(';base64') if head.startswith('data:') else ''
        if not sep or not head.endswith(';base64') or mime not in FORMATS:
            raise ValueError(f'images[{i}] must be a base64 data URL of PNG, JPEG, WebP or GIF '
                             '(data:image/png;base64,...); remote URLs are not fetched')
    return urls


def decode(urls):
    """RGB images in request order; ValueError names the first bad image."""
    from PIL import Image, UnidentifiedImageError
    out = []
    for i, url in enumerate(check(urls)):
        head, _, payload = url.partition(',')
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(f'images[{i}] is not valid base64') from None
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f'images[{i}] is larger than {MAX_IMAGE_BYTES // 2**20} MB')
        try:
            image = Image.open(io.BytesIO(data))
            if image.width * image.height > MAX_PIXELS:
                raise ValueError(f'images[{i}] has more than {MAX_PIXELS} pixels')
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError(f'images[{i}] could not be decoded as an image') from None
        if image.format != FORMATS[head[5:].removesuffix(';base64')]:
            raise ValueError(f'images[{i}] content is {image.format}, not what its data URL declares')
        out.append(image.convert('RGB'))
    return out
