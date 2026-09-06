import os, math, random, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

BASE_DIR   = r'd:\traffic-sign-recognition\ml-training'
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'vn_signs')
SIGN_SIZE  = 128
N_IMAGES   = 250
RANDOM_SEED = 42

RED    = (220, 25, 25)
YELLOW = (255, 212, 0)
WHITE  = (255, 255, 255)
BLACK  = (20, 20, 20)
BLUE   = (20, 85, 195)

def new_canvas(bg=WHITE):
    img = Image.new('RGB', (SIGN_SIZE, SIGN_SIZE), bg)
    return img, ImageDraw.Draw(img)

def draw_warning_triangle(draw, margin=10, yellow=YELLOW):
    cx = SIGN_SIZE // 2
    # Outer red triangle
    top = (cx, margin)
    left = (margin, SIGN_SIZE - margin)
    right = (SIGN_SIZE - margin, SIGN_SIZE - margin)
    draw.polygon([top, left, right], fill=RED)

    # Inner yellow triangle
    bw = 14
    t_top = (cx, margin + int(bw * 1.6))
    t_left = (margin + int(bw * 1.2), SIGN_SIZE - margin - int(bw * 0.7))
    t_right = (SIGN_SIZE - margin - int(bw * 1.2), SIGN_SIZE - margin - int(bw * 0.7))
    draw.polygon([t_top, t_left, t_right], fill=yellow)

def draw_prohibitory_circle(draw, margin=8, fill=WHITE):
    draw.ellipse([margin, margin, SIGN_SIZE - margin, SIGN_SIZE - margin], fill=RED)
    bw = 14
    draw.ellipse([margin + bw, margin + bw, SIGN_SIZE - margin - bw, SIGN_SIZE - margin - bw], fill=fill)

def draw_mandatory_circle(draw, margin=8):
    draw.ellipse([margin, margin, SIGN_SIZE - margin, SIGN_SIZE - margin], fill=BLUE, outline=WHITE, width=2)

def arrow_poly(cx, cy, sz, direction='up'):
    hw = sz // 3; hl = sz // 2; sw = hw // 2
    if direction == 'up':
        return [(cx, cy-hl), (cx+hw, cy), (cx+sw, cy), (cx+sw, cy+hl), (cx-sw, cy+hl), (cx-sw, cy), (cx-hw, cy)]
    elif direction == 'down':
        p = arrow_poly(cx, cy, sz, 'up'); return [(x, 2*cy - y) for x, y in p]
    elif direction == 'right':
        return [(cx+hl, cy), (cx, cy-hw), (cx, cy-sw), (cx-hl, cy-sw), (cx-hl, cy+sw), (cx, cy+sw), (cx, cy+hw)]
    elif direction == 'left':
        p = arrow_poly(cx, cy, sz, 'right'); return [(2*cx - x, y) for x, y in p]

# ── Sign Drawers ─────────────────────────────────────────────────────────────

# 0: W.201a - Chỗ ngoặt nguy hiểm bên trái
def d0(draw, margin=10, yellow=YELLOW):
    draw_warning_triangle(draw, margin, yellow)
    cx, cy = SIGN_SIZE // 2, int(SIGN_SIZE * 0.65)
    r = 18
    for t in range(0, 91, 3):
        rad = math.radians(t)
        x = int(cx - r + r * math.cos(rad))
        y = int(cy - r * math.sin(rad))
        draw.ellipse([x-3, y-3, x+3, y+3], fill=BLACK)
    draw.polygon([(cx-r-6, cy-2), (cx-r+5, cy-9), (cx-r+5, cy+5)], fill=BLACK)

# 1: W.201b - Chỗ ngoặt nguy hiểm bên phải
def d1(draw, margin=10, yellow=YELLOW):
    draw_warning_triangle(draw, margin, yellow)
    cx, cy = SIGN_SIZE // 2, int(SIGN_SIZE * 0.65)
    r = 18
    for t in range(0, 91, 3):
        rad = math.radians(t)
        x = int(cx + r - r * math.cos(rad))
        y = int(cy - r * math.sin(rad))
        draw.ellipse([x-3, y-3, x+3, y+3], fill=BLACK)
    draw.polygon([(cx+r+6, cy-2), (cx+r-5, cy-9), (cx+r-5, cy+5)], fill=BLACK)

# 2: W.207a - Đường hai chiều
def d2(draw, margin=10, yellow=YELLOW):
    draw_warning_triangle(draw, margin, yellow)
    cx, cy = SIGN_SIZE // 2, int(SIGN_SIZE * 0.60)
    # Left arrow down, right arrow up (standard W.207a)
    draw.polygon(arrow_poly(cx - 11, cy, 34, 'down'), fill=BLACK)
    draw.polygon(arrow_poly(cx + 11, cy, 34, 'up'), fill=BLACK)

# 3: W.208 - Giao nhau với đường sắt có rào chắn
def d3(draw, margin=10, yellow=YELLOW):
    draw_warning_triangle(draw, margin, yellow)
    cx, cy = SIGN_SIZE // 2, int(SIGN_SIZE * 0.62)
    draw.line([(cx-20, cy-18), (cx+20, cy+18)], fill=BLACK, width=5)
    draw.line([(cx+20, cy-18), (cx-20, cy+18)], fill=BLACK, width=5)
    draw.rectangle([cx-24, cy-3, cx+24, cy+3], fill=BLACK)

# 4: W.224 - Đường trơn trượt
def d4(draw, margin=10, yellow=YELLOW):
    draw_warning_triangle(draw, margin, yellow)
    cx, cy = SIGN_SIZE // 2, int(SIGN_SIZE * 0.63)
    draw.rectangle([cx-16, cy-6, cx+16, cy+6], fill=BLACK)
    draw.rectangle([cx-14, cy-17, cx+14, cy-7], fill=BLACK)
    draw.ellipse([cx-18, cy+4, cx-8, cy+14], fill=BLACK)
    draw.ellipse([cx+8, cy+4, cx+18, cy+14], fill=BLACK)
    for i in range(3):
        x = cx - 18 + i * 14
        draw.arc([x, cy+14, x+14, cy+22], 0, 180, fill=BLACK, width=2)

# 5: W.245 - Trẻ em
def d5(draw, margin=10, yellow=YELLOW):
    draw_warning_triangle(draw, margin, yellow)
    cx, cy = SIGN_SIZE // 2, int(SIGN_SIZE * 0.61)
    draw.ellipse([cx-7, cy-25, cx+5, cy-13], fill=BLACK)
    draw.line([(cx-1, cy-13), (cx-1, cy+2)], fill=BLACK, width=4)
    draw.line([(cx-12, cy-7), (cx+10, cy-7)], fill=BLACK, width=3)
    draw.line([(cx-1, cy+2), (cx-10, cy+18)], fill=BLACK, width=3)
    draw.line([(cx-1, cy+2), (cx+9, cy+18)], fill=BLACK, width=3)

# 6: P.102 - Đường cấm (hoặc Cấm đi ngược chiều)
def d6(draw, margin=8, **kwargs):
    draw.ellipse([margin, margin, SIGN_SIZE - margin, SIGN_SIZE - margin], fill=RED)
    cy = SIGN_SIZE // 2
    draw.rectangle([margin + 12, cy - 8, SIGN_SIZE - margin - 12, cy + 8], fill=WHITE)

# 7: P.103a - Cấm xe ô tô
def d7(draw, margin=8, **kwargs):
    draw_prohibitory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2
    draw.rectangle([cx-20, cy-6, cx+20, cy+8], fill=BLACK)
    draw.rectangle([cx-14, cy-16, cx+14, cy-6], fill=BLACK)
    draw.ellipse([cx-18, cy+5, cx-8, cy+15], fill=BLACK)
    draw.ellipse([cx+8, cy+5, cx+18, cy+15], fill=BLACK)
    # Red diagonal slash
    draw.line([(margin+14, margin+14), (SIGN_SIZE-margin-14, SIGN_SIZE-margin-14)], fill=RED, width=7)

# 8: P.104 - Cấm xe mô tô
def d8(draw, margin=8, **kwargs):
    draw_prohibitory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2
    draw.ellipse([cx-22, cy-3, cx-8, cy+11], outline=BLACK, width=3)
    draw.ellipse([cx+8, cy-3, cx+22, cy+11], outline=BLACK, width=3)
    draw.line([(cx-15, cy+4), (cx+15, cy+4)], fill=BLACK, width=4)
    draw.line([(cx-5, cy-11), (cx+5, cy+4)], fill=BLACK, width=3)
    draw.ellipse([cx-5, cy-16, cx+5, cy-6], fill=BLACK)
    draw.line([(margin+14, margin+14), (SIGN_SIZE-margin-14, SIGN_SIZE-margin-14)], fill=RED, width=7)

# 9: P.123 - Cấm vượt / Cấm rẽ
def d9(draw, margin=8, **kwargs):
    draw_prohibitory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2
    draw.rectangle([cx-24, cy-5, cx-4, cy+7], fill=BLACK)
    draw.rectangle([cx+4, cy-5, cx+24, cy+7], outline=BLACK, width=2)
    draw.line([(margin+14, margin+14), (SIGN_SIZE-margin-14, SIGN_SIZE-margin-14)], fill=RED, width=7)

# Helper for speed signs
def _speed(draw, txt, margin=8):
    draw_prohibitory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2 + 1
    font = None
    for fn in ['arialbd.ttf', 'arial.ttf', 'seguisb.ttf']:
        try:
            font = ImageFont.truetype(fn, 52)
            break
        except:
            pass
    if not font:
        font = ImageFont.load_default()

    bb = draw.textbbox((0, 0), txt, font=font)
    tw = bb[2] - bb[0]; th = bb[3] - bb[1]
    draw.text((cx - tw // 2, cy - th // 2 - 2), txt, fill=BLACK, font=font)

# 10: P.127-40 - Tốc độ tối đa 40 km/h
def d10(draw, margin=8, **kwargs): _speed(draw, '40', margin)

# 11: P.127-60 - Tốc độ tối đa 60 km/h
def d11(draw, margin=8, **kwargs): _speed(draw, '60', margin)

# 12: R.301a - Hướng đi phải theo: Đi thẳng
def d12(draw, margin=8, **kwargs):
    draw_mandatory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2
    draw.polygon(arrow_poly(cx, cy + 6, 48, 'up'), fill=WHITE)

# 13: R.302a - Hướng đi phải theo: Rẽ phải
def d13(draw, margin=8, **kwargs):
    draw_mandatory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2
    draw.polygon(arrow_poly(cx - 6, cy, 48, 'right'), fill=WHITE)

# 14: R.303 - Giao thông hai chiều (Mandatory)
def d14(draw, margin=8, **kwargs):
    draw_mandatory_circle(draw, margin)
    cx, cy = SIGN_SIZE // 2, SIGN_SIZE // 2
    draw.polygon(arrow_poly(cx + 12, cy, 38, 'up'), fill=WHITE)
    draw.polygon(arrow_poly(cx - 12, cy, 38, 'down'), fill=WHITE)

DRAWERS = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14]
NAMES   = [
    'W.201a', 'W.201b', 'W.207a', 'W.208', 'W.224', 'W.245',
    'P.102', 'P.103a', 'P.104', 'P.123', 'P.127-40', 'P.127-60',
    'R.301a', 'R.302a', 'R.303'
]

def augment(img):
    # Rotate slightly
    angle = random.uniform(-14, 14)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=WHITE)

    # Scale / Crop
    w, h = img.size
    sc = random.uniform(0.85, 1.0)
    nw, nh = int(w * sc), int(h * sc)
    l = random.randint(0, w - nw)
    t = random.randint(0, h - nh)
    img = img.crop((l, t, l + nw, t + nh)).resize((w, h), Image.BILINEAR)

    # Brightness & Contrast
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    # Optional slight blur
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.4, 1.0)))

    # Pixel noise
    arr = np.array(img, dtype=np.int16) + np.random.randint(-15, 15, np.array(img).shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Generating {len(DRAWERS)} classes x {N_IMAGES} images (QCVN 41:2019/BGTVT standard)...')

    yellow_shades = [
        (255, 212, 0),   # Standard traffic yellow
        (255, 222, 10),  # Bright yellow
        (250, 200, 0),   # Deep yellow
        (245, 195, 0),   # Golden yellow
    ]

    for cid, (dr, nm) in enumerate(zip(DRAWERS, NAMES)):
        cdir = os.path.join(OUTPUT_DIR, str(cid))
        os.makedirs(cdir, exist_ok=True)
        is_warning = (cid < 6)

        for i in range(N_IMAGES):
            # Vary margin
            m = random.randint(6, 16) if i >= 10 else 10
            y_tone = random.choice(yellow_shades) if is_warning else YELLOW

            img, draw = new_canvas()
            if is_warning:
                dr(draw, margin=m, yellow=y_tone)
            else:
                dr(draw, margin=m)

            if i >= 10:
                img = augment(img)

            img.resize((32, 32), Image.BILINEAR).save(os.path.join(cdir, f'{i:04d}.png'))

        print(f'  [{cid:2d}] {nm}: {N_IMAGES} done')

    print(f'Done! {len(DRAWERS)*N_IMAGES} total images.')

if __name__ == '__main__':
    main()
