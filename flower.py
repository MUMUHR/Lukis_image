import os
import sys
import time
import math
import glob
import turtle
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw


QUALITY_PRESETS = {
    "ULTRA": {
        "name": "Ultra Akurat (HD - High Detail)",
        "max_width": 640,
        "num_colors": 28,
        "blur_amount": 3,
        "min_contour_area": 8,
        "detail_level": 0.0025,
        "edge_threshold": (0.60, 1.35),
        "detail_layer": True,
    },
    "BALANCED": {
        "name": "Seimbang (Balanced - Rekomendasi)",
        "max_width": 520,
        "num_colors": 20,
        "blur_amount": 5,
        "min_contour_area": 18,
        "detail_level": 0.004,
        "edge_threshold": (0.65, 1.35),
        "detail_layer": True,
    },
    "FAST": {
        "name": "Cepat (Fast Preview)",
        "max_width": 380,
        "num_colors": 12,
        "blur_amount": 7,
        "min_contour_area": 45,
        "detail_level": 0.008,
        "edge_threshold": (0.70, 1.30),
        "detail_layer": False,
    },
}

CONTENT_PRESETS = {
    "PHOTO": {
        "name": "Foto Umum / Bunga / Pemandangan",
        "blur_sigma": 40,
        "edge_weight": 0.65,
    },
    "PORTRAIT": {
        "name": "Wajah / Potret Manusia",
        "num_colors_boost": 4,
        "min_contour_area_factor": 0.6,
        "blur_sigma": 35,
        "edge_weight": 0.70,
    },
    "ANIME": {
        "name": "Anime / Ilustrasi / Line Art",
        "num_colors_boost": 6,
        "blur_amount": 3,
        "blur_sigma": 25,
        "edge_weight": 0.50,
        "detail_level": 0.002,
    },
}

CANVAS_MAX = 850
DRAW_SPEED = 0               
SHADOW_LAYER_ENABLED = True 




def load_image(path: str) -> np.ndarray:
    """Membuka file gambar menggunakan OpenCV dan memeriksa keabsahan file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di path: '{path}'")
    
    img = cv2.imread(path)
    if img is None:
        
        try:
            pil_img = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            raise ValueError(f"Tidak dapat membaca format gambar di: '{path}'")
            
    return img



def preprocess_image(
    img: np.ndarray,
    max_width: int = 520,
    blur_amount: int = 5,
    sigma_color: float = 40.0
) -> np.ndarray:
    
    h, w = img.shape[:2]
    longest_side = max(h, w)
    scale = max_width / float(longest_side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

 
    gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
    sharpened = cv2.addWeighted(resized, 1.25, gaussian, -0.25, 0)

  
    d = max(3, blur_amount * 2 + 1)
    smoothed = cv2.bilateralFilter(sharpened, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_color)
    return smoothed



def quantize_colors_lab(
    img_bgr: np.ndarray,
    num_colors: int = 20
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
  
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    data = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, labels, _ = cv2.kmeans(
        data, num_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape((h, w))

    
    cluster_colors_rgb = []
    for c in range(num_colors):
        mask = (labels == c)
        if np.any(mask):
            pixels_bgr = img_bgr[mask]
            med_bgr = np.median(pixels_bgr, axis=0)
            # Konversi BGR -> RGB
            r = int(np.clip(med_bgr[2], 0, 255))
            g = int(np.clip(med_bgr[1], 0, 255))
            b = int(np.clip(med_bgr[0], 0, 255))
            cluster_colors_rgb.append((r, g, b))
        else:
            cluster_colors_rgb.append((0, 0, 0))

    return labels, cluster_colors_rgb



def extract_contours(
    labels: np.ndarray,
    cluster_colors_rgb: List[Tuple[int, int, int]],
    min_area: float = 18.0,
    detail_level: float = 0.004
) -> List[Dict]:
  
    h, w = labels.shape[:2]
    layers = []
    num_colors = len(cluster_colors_rgb)

    close_kernel = np.ones((3, 3), np.uint8)
    dilate_kernel = np.ones((2, 2), np.uint8)

    for c in range(num_colors):
        mask = np.uint8(labels == c) * 255
        if cv2.countNonZero(mask) == 0:
            continue

      
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

       
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        color_rgb = cluster_colors_rgb[c]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

           
            perimeter = cv2.arcLength(cnt, True)
            epsilon = max(0.4, detail_level * perimeter)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) >= 3:
                points = [(float(p[0][0]), float(p[0][1])) for p in approx]
                layers.append({
                    "points": points,
                    "color": color_rgb,
                    "area": area,
                    "is_line": False,
                })

    return layers


def extract_detail_region_layers(
    img_bgr: np.ndarray,
    min_area: float = 6.0,
    detail_level: float = 0.002
) -> List[Dict]:
   
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, thresh = cv2.threshold(norm_mag, 65, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = img_bgr.shape[:2]
    detail_layers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:  
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        
        roi = img_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
            
   
        roi_labels, roi_colors = quantize_colors_lab(roi, num_colors=10)
        roi_layers = extract_contours(roi_labels, roi_colors, min_area=min_area, detail_level=detail_level)
        
        for lyr in roi_layers:
           
            offset_pts = [(px + x0, py + y0) for (px, py) in lyr["points"]]
            detail_layers.append({
                "points": offset_pts,
                "color": lyr["color"],
                "area": lyr["area"],
                "is_line": False,
            })

    return detail_layers



def extract_natural_edge_layers(
    img_bgr: np.ndarray,
    edge_threshold: Tuple[float, float] = (0.65, 1.35),
    darken_factor: float = 0.65,
    min_length: float = 10.0
) -> List[Dict]:
    
    h_img, w_img = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Auto-Canny adaptif
    median_val = float(np.median(gray))
    lower = int(max(0, edge_threshold[0] * median_val))
    upper = int(min(255, edge_threshold[1] * median_val))
    if upper <= lower:
        upper = lower + 35

    edges = cv2.Canny(gray, lower, upper)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_layers = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, False)
        if perimeter < min_length:
            continue

        epsilon = max(0.5, 0.0035 * perimeter)
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        if len(approx) < 2:
            continue

        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        
       
        xs = [int(np.clip(p[0], 0, w_img - 1)) for p in pts]
        ys = [int(np.clip(p[1], 0, h_img - 1)) for p in pts]
        sampled_bgr = np.mean([img_bgr[y, x] for x, y in zip(xs, ys)], axis=0)
        

        r = int(np.clip(sampled_bgr[2] * darken_factor, 0, 255))
        g = int(np.clip(sampled_bgr[1] * darken_factor, 0, 255))
        b = int(np.clip(sampled_bgr[0] * darken_factor, 0, 255))

        edge_layers.append({
            "points": pts,
            "color": (r, g, b),
            "area": perimeter,
            "is_line": True,
        })

    return edge_layers


def make_background_layer(img_w: int, img_h: int, avg_color_rgb: Tuple[int, int, int]) -> Dict:
    
    points = [(0.0, 0.0), (float(img_w), 0.0), (float(img_w), float(img_h)), (0.0, float(img_h))]
    return {
        "points": points,
        "color": avg_color_rgb,
        "area": float(img_w * img_h * 100),
        "is_line": False,
    }


def compute_average_color(img_bgr: np.ndarray) -> Tuple[int, int, int]:
    
    b, g, r = cv2.mean(img_bgr)[:3]
    return (int(r), int(g), int(b))


def sort_layers(layers: List[Dict]) -> List[Dict]:
    
    return sorted(layers, key=lambda l: l["area"], reverse=True)


def to_turtle_coords(points: List[Tuple[float, float]], img_w: int, img_h: int) -> List[Tuple[float, float]]:
    
    return [(x - img_w / 2.0, (img_h / 2.0) - y) for (x, y) in points]


def draw_line_with_turtle(t: turtle.Turtle, points: List[Tuple[float, float]], color: Tuple[int, int, int]):
   
    if len(points) < 2:
        return
    r, g, b = color
    t.pencolor(r / 255.0, g / 255.0, b / 255.0)
    t.width(1.2)
    t.penup()
    t.goto(points[0])
    t.pendown()
    for pt in points[1:]:
        t.goto(pt)
    t.penup()


def draw_polygon_with_turtle(
    t: turtle.Turtle,
    points: List[Tuple[float, float]],
    color: Tuple[int, int, int]
):
    
    if len(points) < 3:
        return

    r, g, b = color
    color_norm = (r / 255.0, g / 255.0, b / 255.0)

    t.fillcolor(color_norm)
    t.pencolor(color_norm)
    t.width(1.3)  

    t.penup()
    t.goto(points[0])
    t.pendown()
    t.begin_fill()
    for pt in points[1:]:
        t.goto(pt)
    t.goto(points[0])
    t.end_fill()
    t.penup()


def animate_drawing(
    layers: List[Dict],
    img_w: int,
    img_h: int,
    title: str = "MUNAR Painting PEAK"
) -> turtle.Screen:
 
    screen = turtle.Screen()
    screen.title(title)
    

    canvas_w = min(img_w, CANVAS_MAX)
    canvas_h = min(img_h, CANVAS_MAX)
    screen.setup(width=canvas_w + 50, height=canvas_h + 50)
    screen.setworldcoordinates(-img_w / 2.0, -img_h / 2.0, img_w / 2.0, img_h / 2.0)

    screen.bgcolor("white")
    screen.tracer(0, 0)

    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.penup()

    total = len(layers)
    print(f"\n[INFO] Mulai melukis {total} layer presisi...")
    
    start_time = time.time()
    last_update_time = time.time()
    
   
    for i, layer in enumerate(layers, start=1):
        pts = to_turtle_coords(layer["points"], img_w, img_h)
        area = layer.get("area", 0)

        if layer.get("is_line"):
            draw_line_with_turtle(t, pts, layer["color"])
        else:
            draw_polygon_with_turtle(t, pts, layer["color"])

   
        now = time.time()
        is_big_layer = (i <= 25) or (area > (img_w * img_h * 0.02))
        
        if is_big_layer:
            screen.update()
            time.sleep(0.015)
            last_update_time = now
        elif (now - last_update_time > 0.04) or (i % 20 == 0) or (i == total):
            screen.update()
            last_update_time = now

        
        if i % 25 == 0 or i == total:
            percent = int((i / total) * 100)
            bar_len = 25
            filled_len = int(bar_len * i // total)
            bar = '█' * filled_len + '░' * (bar_len - filled_len)
            print(f"\r  Progress: [{bar}] {percent:3d}% ({i}/{total} layers)", end="", flush=True)

    screen.update()
    duration = time.time() - start_time
    print(f"\n[SELESAI] Lukisan selesai dalam {duration:.1f} detik!")
    print("Klik jendela lukisan untuk keluar.")
    return screen



def run_photo_pipeline(
    image_path: str,
    quality_mode: str = "ULTRA",
    content_type: str = "PHOTO"
) -> bool:
   
    q_cfg = QUALITY_PRESETS.get(quality_mode, QUALITY_PRESETS["BALANCED"]).copy()
    c_cfg = CONTENT_PRESETS.get(content_type, CONTENT_PRESETS["PHOTO"])

    # Terapkan modifikasi dari content preset
    max_w = q_cfg["max_width"]
    num_cols = q_cfg["num_colors"] + c_cfg.get("num_colors_boost", 0)
    blur_amt = c_cfg.get("blur_amount", q_cfg["blur_amount"])
    blur_sig = c_cfg.get("blur_sigma", 40.0)
    min_area = q_cfg["min_contour_area"] * c_cfg.get("min_contour_area_factor", 1.0)
    detail_lvl = c_cfg.get("detail_level", q_cfg["detail_level"])
    edge_thresh = q_cfg["edge_threshold"]
    edge_weight = c_cfg.get("edge_weight", 0.65)

    print(f"\n--- Memproses Gambar: {os.path.basename(image_path)} ---")
    print(f"Mode Kualitas: {q_cfg['name']}")
    print(f"Jenis Gambar : {c_cfg['name']}")

    img = load_image(image_path)

    processed = preprocess_image(img, max_width=max_w, blur_amount=blur_amt, sigma_color=blur_sig)
    img_h, img_w = processed.shape[:2]
    print(f"Resolusi Kanvas: {img_w}x{img_h} piksel")


    print(f"Mengekstrak {num_cols} palet warna presisi (CIE-Lab Space)...")
    labels, colors_rgb = quantize_colors_lab(processed, num_colors=num_cols)


    print("Menganalisis poligon dan bentuk kontur...")
    main_layers = extract_contours(labels, colors_rgb, min_area=min_area, detail_level=detail_lvl)
    print(f"Ditemukan {len(main_layers)} poligon warna utama.")


    avg_color = compute_average_color(processed)
    bg_layer = make_background_layer(img_w, img_h, avg_color)


    ordered_polys = sort_layers(main_layers)

   
    detail_polys = []
    if q_cfg.get("detail_layer", True):
        print("Mendeteksi area dengan detail tinggi...")
        detail_polys = extract_detail_region_layers(processed, min_area=max(4.0, min_area * 0.4), detail_level=detail_lvl * 0.7)
        if detail_polys:
            print(f"Menambahkan {len(detail_polys)} poligon mikro untuk area detail.")
            detail_polys = sort_layers(detail_polys)

    
    edge_lines = []
    if SHADOW_LAYER_ENABLED:
        print("Mengekstrak garis detail alami (Natural Edge Lines)...")
        edge_lines = extract_natural_edge_layers(
            processed,
            edge_threshold=edge_thresh,
            darken_factor=edge_weight,
            min_length=10.0
        )
        print(f"Ditemukan {len(edge_lines)} garis aksen detail.")

    all_layers = [bg_layer] + ordered_polys + detail_polys + edge_lines

    if not all_layers:
        print("[ERROR] Tidak ada layer yang bisa digambar. Coba ubah mode kualitas.")
        return False

    animate_drawing(all_layers, img_w, img_h, title=f"Painting: {os.path.basename(image_path)}")
    return True



def make_tulip_petal(cx: float, cy: float, width: float, height: float, rotation_deg: float = 0) -> List[Tuple[float, float]]:
    pts = []
    rot = math.radians(rotation_deg)
    n = 16
    for i in range(n + 1):
        t = i / float(n)
        angle = math.pi * t
        x = width * math.cos(angle)
        y = -height * math.sin(angle)
        pts.append((x, y))
    pts.append((0.0, height * 1.3))
    rotated = []
    for (x, y) in pts:
        rx = x * math.cos(rot) - y * math.sin(rot)
        ry = x * math.sin(rot) + y * math.cos(rot)
        rotated.append((cx + rx, cy + ry))
    return rotated


def make_leaf(cx: float, cy: float, length: float, width: float, rotation_deg: float = 0) -> List[Tuple[float, float]]:
    pts = []
    n = 14
    for i in range(n + 1):
        t = i / float(n)
        x = t * length
        y = width * math.sin(math.pi * t)
        pts.append((x, y))
    for i in range(n + 1):
        t = i / float(n)
        x = (1.0 - t) * length
        y = -width * 0.4 * math.sin(math.pi * (1.0 - t))
        pts.append((x, y))
    rot = math.radians(rotation_deg)
    rotated = []
    for (x, y) in pts:
        rx = x * math.cos(rot) - y * math.sin(rot)
        ry = x * math.sin(rot) + y * math.cos(rot)
        rotated.append((cx + rx, cy + ry))
    return rotated


def polygon_area(points: List[Tuple[float, float]]) -> float:
    pts = np.array(points, dtype=np.float32).reshape((-1, 1, 2))
    return abs(cv2.contourArea(pts))


def build_manual_bouquet() -> List[Dict]:
    layers = []

    def add(points, color):
        layers.append({"points": points, "color": color, "area": polygon_area(points), "is_line": False})

    # Background
    bg_pts = [(-200, -200), (200, -200), (200, 200), (-200, 200)]
    layers.append({"points": bg_pts, "color": (250, 248, 245), "area": 160000.0, "is_line": False})

    # Pembungkus buket
    wrap = [(-90, -180), (90, -180), (130, -40), (0, 40), (-130, -40)]
    add(wrap, (196, 164, 118))

    wrap_shadow = [(-60, -160), (60, -160), (90, -60), (0, 0), (-90, -60)]
    add(wrap_shadow, (168, 136, 92))

    # Daun
    leaf_positions = [
        (-40, 20, 90, 22, 100), (30, 10, 100, 24, 70), (-10, 30, 80, 20, 95),
        (60, 0, 70, 18, 55), (-70, -10, 70, 18, 120),
    ]
    for (x, y, length, width, rot) in leaf_positions:
        add(make_leaf(x, y, length, width, rot), (74, 124, 66))


    tulip_positions = [
        (-70, 140), (-30, 165), (10, 150), (50, 170),
        (85, 130), (-50, 110), (30, 115),
    ]
    for (x, y) in tulip_positions:
        add(make_tulip_petal(x, y, 22, 40, 0), (255, 235, 238))
        add(make_tulip_petal(x, y + 6, 16, 24, 0), (233, 135, 145))
        add(make_tulip_petal(x, y + 10, 10, 15, 0), (210, 85, 105))


    ribbon = [(-15, -100), (15, -100), (18, -115), (-18, -115)]
    add(ribbon, (245, 245, 240))

    tag = [(-10, -150), (35, -150), (35, -125), (-10, -125)]
    add(tag, (240, 235, 225))

    return layers


def run_manual_bouquet():
    print("\nMenyusun layer buket tulip artistik...")
    layers = build_manual_bouquet()
    ordered = sort_layers(layers)
    animate_drawing(ordered, img_w=400, img_h=400, title="Turtle Painting - Buket Tulip (Manual)")


def generate_sample_image(save_path: str = "sample_landscape.png", size: Tuple[int, int] = (520, 620)) -> str:
    w, h = size
    img = Image.new("RGB", (w, h), (235, 240, 245))
    draw = ImageDraw.Draw(img)


    for y in range(0, int(h * 0.55)):
        t = y / (h * 0.55)
        color = (int(140 + t * 90), int(180 + t * 50), int(235 - t * 40))
        draw.line([(0, y), (w, y)], fill=color)


    draw.ellipse([w * 0.65, h * 0.12, w * 0.85, h * 0.28], fill=(255, 215, 110))


    draw.polygon([(0, h * 0.50), (w * 0.35, h * 0.38), (w * 0.75, h * 0.48), (w, h * 0.42), (w, h), (0, h)],
                 fill=(95, 140, 115))

    draw.polygon([(0, h * 0.60), (w * 0.55, h * 0.48), (w, h * 0.62), (w, h), (0, h)],
                 fill=(60, 115, 75))
    

    draw.rectangle([w * 0.22, h * 0.58, w * 0.26, h * 0.78], fill=(85, 55, 35))
    draw.ellipse([w * 0.12, h * 0.38, w * 0.36, h * 0.62], fill=(45, 95, 50))
    draw.ellipse([w * 0.16, h * 0.42, w * 0.32, h * 0.56], fill=(70, 130, 65))

    img.save(save_path)
    print(f"Gambar contoh dibuat: {save_path}")
    return save_path


def run_sample_image_mode():
    path = generate_sample_image()
    run_photo_pipeline(path, quality_mode="ULTRA", content_type="PHOTO")


def get_available_local_images() -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    found = []
    for ext in exts:
        found.extend(glob.glob(ext))
    # Filter file sample jika ada
    return sorted(list(set(found)))


def ask_for_image_path() -> Optional[str]:
    local_images = get_available_local_images()
    
    print("\nPilih sumber gambar:")
    if local_images:
        print("  Ditemukan gambar di folder kerja:")
        for idx, img_file in enumerate(local_images, start=1):
            print(f"    {idx}) {img_file}")
        print("    0) Masukkan path gambar lain / buka file dialog")
        choice = input(f"Pilih (0-{len(local_images)}): ").strip()
        
        if choice.isdigit():
            c_int = int(choice)
            if 1 <= c_int <= len(local_images):
                return local_images[c_int - 1]

    print("\nMasukkan path gambar lengkap (contoh: D:\\Foto\\bunga.png)")
    print("Atau tekan Enter langsung untuk membuka jendela File Dialog.")
    user_input = input("Path: ").strip().strip('"')

    if user_input:
        if os.path.isfile(user_input):
            return user_input
        print(f"[PERINGATAN] File tidak ditemukan di: '{user_input}'")
        return None


    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            title="Pilih Gambar untuk Dilukis",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        root.destroy()
        if chosen and os.path.isfile(chosen):
            return chosen
        print("[INFO] Tidak ada file yang dipilih.")
        return None
    except Exception as e: 
        print(f"[INFO] File dialog tidak tersedia ({e}). Silakan ketik path file secara manual.")
        return None


def ask_quality_mode() -> str:
    print("\nPilih Tingkat Kualitas / Presisi:")
    print("  1) Ultra Akurat (HD - Paling mirip gambar kek asli, sangat detail)")
    print("  2) Seimbang (Balanced - Rekomendasi, cepat & bagus)")
    print("  3) Cepat (Fast Preview)")
    choice = input("Pilih kualitas (1/2/3) [Default 1]: ").strip()
    return {"1": "ULTRA", "2": "BALANCED", "3": "FAST"}.get(choice, "ULTRA")


def ask_content_type() -> str:
    print("\nPilih Jenis Objek Gambar:")
    print("  1) Foto Umum / Bunga / Pemandangan / Benda")
    print("  2) Wajah / poto Manusia")
    print("  3) Anime / Ilustrasi atau lain")
    choice = input("Pilih jenis (1/2/3) [Default 1]: ").strip()
    return {"1": "PHOTO", "2": "PORTRAIT", "3": "ANIME"}.get(choice, "PHOTO")


def run_own_photo_mode():
    path = ask_for_image_path()
    if not path:
        print("[INFO] Batal memproses gambar.")
        return
        
    quality = ask_quality_mode()
    content = ask_content_type()
    
    run_photo_pipeline(path, quality_mode=quality, content_type=content)

def show_menu() -> str:
    print("=" * 60)
    print("        TURTLE PAINTING EFFECT - ULTRA ACCURATE v4        ")
    print("=" * 60)
    print("  1) Gambar Manual (Buket Tulip Vektor dari Kode)")
    print("  2) Gambar Contoh (Lanskap Pemandangan Otomatsis)")
    print("  3) Lukis Foto Sendiri (Bunga / Anime / dlll)")
    print("  4) Keluar")
    print("=" * 60)
    choice = input("Pilih menu (1/2/3/4): ").strip()
    return choice


def main():
    while True:
        choice = show_menu()
        if choice == "1":
            run_manual_bouquet()
            break
        elif choice == "2":
            run_sample_image_mode()
            break
        elif choice == "3":
            run_own_photo_mode()
            break
        elif choice == "4" or choice.lower() in ("q", "exit"):
            print("Keluar dari program. Sampai jumpa!")
            sys.exit(0)
        else:
            print("Pilihan tidak valid, silakan coba lagi.\n")

    try:
        screen = turtle.Screen()
        screen.exitonclick()
    except Exception:
        pass


if __name__ == "__main__":
    main()