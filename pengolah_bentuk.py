# -*- coding: utf-8 -*-
"""
PENGOLAH BENTUK

Alur:
1. Cari garis gambar di foto
2. Telusuri garisnya
3. Diubah jadi gambar 64x64px
"""

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

VERSI_PENGOLAH = 4

UKURAN_MODEL = 64          # hasil akhir tiap bentuk 64x64px
MARGIN_HASIL = 12          # margin
JENDELA_LATAR = 0.30       # untuk memperkiraan cahaya/bayangan (dibanding bagian background)
JENDELA_BESAR = 0.60       # dipakai kalau pada percobaan pertama tidak ada bentuk yang ditemukan
JENDELA_PENUH = 1.00       # percobaan terakhir
FAKTOR_BATAS = (0.5, 1.0)  # batas goresan
TAHAP_GABUNG = (0.012, 0.02, 0.03, 0.045, 0.065, 0.09)   # jarak penyambungan garis putus-putus
BUANG_BINTIK = 0.00004     # potongan garis yang lebih kecil dari ini dibuang
LUAS_MIN_REL = 0.08        # bentuk lebih kecil dari ini (dibanding yang terbesar) dibuang
MARGIN_TEPI = 0.01         # objek pada tepi foto dibuang
UKURAN_MIN = 0.05          # ukuran objek minimal
UKURAN_MAX = 0.95          # ukuran objek maksimal
MAKS_BENTUK = 12           # jumlah bentuk maksimal dalam satu foto
TAJAM_MIN = 5.0            # batas minimum gradien perubahan warna
TOLERANSI_WARNA = 0.06     # selisih warna background yang masih diterima
TERANG_MIN = 0.45          # batas bawah kecerahan background dibanding kertas
TERANG_MAX = 1.35          # batas atas kecerahan background dibanding kertas
SEBARAN_LATAR = 0.35       # background di sekeliling bentuk harus seragam
RING_LOKAL = 0.03          # jarak zona di sekeliling objek untuk memastikan bahwa bg di sekitar objek seragam
LUBANG_MIN = 0.2           # garis dianggap tertutup kalau bagian dalamnya lumayan luas (>= 20%)
PADAT_MIN = 0.25           # objek dianggap berisi kalau setelah dikikis masih sisa minimal 25%
KEPADATAN_MIN = 0.92       # agar model tidak scan tangan, jadi langsung dibuang
HALUS_SUDUT = 0.03         # garis dihaluskan agar lebih rapi/lurus
RASIO_MAX = 6.0            # untuk menyaring bena lain yang terfoto di atas kertas


def otsu(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    jumlah_semua = np.dot(np.arange(256), hist)
    jumlah_b, bobot_b, terbaik, batas = 0.0, 0, 0.0, 0
    for t in range(256):
        bobot_b += hist[t]
        if bobot_b == 0:
            continue
        bobot_f = total - bobot_b
        if bobot_f == 0:
            break
        jumlah_b += t * hist[t]
        rata_b = jumlah_b / bobot_b
        rata_f = (jumlah_semua - jumlah_b) / bobot_f
        selisih = bobot_b * bobot_f * (rata_b - rata_f) ** 2
        if selisih > terbaik:
            terbaik, batas = selisih, t
    return batas


def warna_kertas(arr):
    """Mencari warna kertas dan bagian pinggir luar diabaikan."""
    arr = arr.astype('float32')
    lum = arr.mean(axis=2)
    h, w = lum.shape
    rata = ndi.uniform_filter(lum, size=7)
    sebaran = np.sqrt(np.clip(ndi.uniform_filter(lum * lum, size=7) - rata * rata, 0, None))
    batas = max(4.0, 2.0 * float(np.percentile(sebaran, 20)))
    mulus = ndi.binary_erosion(sebaran < batas, structure=np.ones((3, 3)), iterations=3)
    label, n = ndi.label(mulus)
    t = max(2, int(min(h, w) * 0.03))
    if n > 0:
        luas = np.bincount(label.ravel(), minlength=n + 1)
        cincin = np.zeros((h, w), dtype=bool)
        cincin[:t], cincin[-t:], cincin[:, :t], cincin[:, -t:] = True, True, True, True
        sentuh = np.bincount(label[cincin & (label > 0)], minlength=n + 1)
        sentuh[0] = 0
        sentuh[luas < 0.05 * h * w] = 0
        if sentuh.max() > 0:
            pilih = int(np.argmax(sentuh))
        else:
            luas[0] = 0
            pilih = int(np.argmax(luas))
        if luas[pilih] >= 0.05 * h * w:
            return np.median(arr[label == pilih], axis=0).astype('float32')
    tepi = np.concatenate([arr[:t].reshape(-1, 3), arr[-t:].reshape(-1, 3),
                           arr[:, :t].reshape(-1, 3), arr[:, -t:].reshape(-1, 3)])
    return np.median(tepi, axis=0).astype('float32')


def tinta_gelap(arr, kertas):
    """True kalau bentuk lebih gelap dari kertas, false kalau lebih terang."""
    lum = arr.mean(axis=2)
    k = float(kertas.mean())
    p_rendah, p_tinggi = np.percentile(lum, [0.5, 99.5])
    return (k - p_rendah) >= (p_tinggi - k)


def sinyal_goresan(arr, sisi_foto, jendela, gelap_dulu):
    """Mencari garis tinta dengan menyaring cahaya, dimana gradasi bayangan dianggap bagian background."""
    h0, w0 = arr.shape[:2]
    ukuran = max(15, int(sisi_foto * jendela)) | 1
    total = np.zeros((h0, w0), dtype='float32')
    for c in range(3):
        kanal = arr[:, :, c]
        if gelap_dulu:
            latar = ndi.grey_closing(kanal, size=(ukuran, ukuran))
            latar = ndi.uniform_filter(latar, size=ukuran // 2 | 1)
            selisih = np.clip(latar - kanal, 0, None)
        else:
            latar = ndi.grey_opening(kanal, size=(ukuran, ukuran))
            latar = ndi.uniform_filter(latar, size=ukuran // 2 | 1)
            selisih = np.clip(kanal - latar, 0, None)
        total += selisih ** 2
    return np.sqrt(total)


def latar_mirip_kertas(asli, label, i, mask, sl, kertas, sisi_foto):
    h0, w0 = asli.shape[:2]
    jarak_ring = max(8, int(sisi_foto * RING_LOKAL))
    p = jarak_ring + 4
    y0, y1 = max(sl[0].start - p, 0), min(sl[0].stop + p, h0)
    x0, x1 = max(sl[1].start - p, 0), min(sl[1].stop + p, w0)
    blob = label[y0:y1, x0:x1] == i
    goresan = mask[y0:y1, x0:x1] & blob
    if goresan.sum() == 0:
        return True
    semua_tinta = ndi.binary_dilation(mask[y0:y1, x0:x1], iterations=2)
    jarak = ndi.distance_transform_edt(~goresan)
    cincin = (jarak > 3) & (jarak <= jarak_ring) & ~semua_tinta
    if cincin.sum() < 30:
        return True
    piksel = asli[y0:y1, x0:x1][cincin].astype('float32')
    warna = np.median(piksel, axis=0)
    kroma_obj = warna / max(float(warna.sum()), 1.0)
    kroma_kertas = kertas / max(float(kertas.sum()), 1.0)
    selisih_warna = float(np.abs(kroma_obj - kroma_kertas).max())
    rasio = (float(warna.mean()) + 1) / (float(kertas.mean()) + 1)
    if not (selisih_warna <= TOLERANSI_WARNA and TERANG_MIN <= rasio <= TERANG_MAX):
        return False
    p10 = np.percentile(piksel, 10, axis=0)
    p90 = np.percentile(piksel, 90, axis=0)
    sebaran = float(np.max((p90 - p10) / (piksel.mean(axis=0) + 1)))
    return sebaran <= SEBARAN_LATAR


def kumpulkan_objek(mask, sisi_foto):
    """Menelusuri dan menyambung goresan secara bertahap hingga membentuk garis yang tertutup."""
    sisa = mask.copy()
    label_akhir = np.zeros(mask.shape, dtype=np.int32)
    objek = []
    for persen in TAHAP_GABUNG:
        if not sisa.any():
            break
        r = max(3, int(sisi_foto * persen))
        gabung = ndi.distance_transform_edt(~sisa) <= r
        label, n = ndi.label(gabung)
        for i, sl in enumerate(ndi.find_objects(label), start=1):
            if sl is None:
                continue
            wilayah = label[sl] == i
            g = sisa[sl] & wilayah
            luas = int(g.sum())
            if luas == 0:
                continue
            ys, xs = np.where(g)
            kotak = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            lubang = int(ndi.binary_fill_holes(wilayah).sum() - wilayah.sum())
            tebal = ndi.binary_erosion(g, structure=np.ones((3, 3)), iterations=max(3, int(sisi_foto * 0.012))).sum()
            if tebal >= PADAT_MIN * luas or lubang >= LUBANG_MIN * kotak:
                nomor = len(objek) + 1
                sub = label_akhir[sl]
                sub[wilayah & (sub == 0)] = nomor
                sisa[sl][g] = False
                pas = (slice(sl[0].start + ys.min(), sl[0].start + ys.max() + 1),
                       slice(sl[1].start + xs.min(), sl[1].start + xs.max() + 1))
                objek.append((nomor, pas))
    return label_akhir, objek


def cembung(s):
    """Mengubah siluet menjadi bentuk cembung agar garis yang putus-putus terlihat lebih rapih,
    dan kembali ke bentuk asli jika prosesnya gagal."""
    ys, xs = np.where(s)
    if len(ys) < 5:
        return s
    try:
        titik = np.stack([xs, ys], axis=1)
        sudut = [tuple(int(v) for v in titik[i]) for i in ConvexHull(titik).vertices]
        kanvas = Image.new('L', (s.shape[1], s.shape[0]), 0)
        ImageDraw.Draw(kanvas).polygon(sudut, fill=1, outline=1)
        return np.array(kanvas).astype(bool)
    except Exception:
        return s


def siluet(g):
    """Membuat bentuk menjadi terisi penuh dan membingkainya dengan area kosong."""
    sisi = max(g.shape)
    g = np.pad(g, max(12, int(sisi * 0.12)))
    tutup = ndi.binary_closing(g, structure=np.ones((3, 3)), iterations=4)
    isi = ndi.binary_fill_holes(tutup)
    if isi.sum() > tutup.sum() * 1.02:
        return isi
    for persen in (0.02, 0.04, 0.07, 0.10):
        r = max(3, int(sisi * persen))
        gemuk = ndi.binary_dilation(g, structure=np.ones((3, 3)), iterations=r)
        isi = ndi.binary_fill_holes(gemuk)
        if isi.sum() > gemuk.sum() * 1.02:
            isi = ndi.binary_erosion(isi, structure=np.ones((3, 3)), iterations=r)
            rapi = ndi.binary_opening(isi, structure=np.ones((3, 3)), iterations=max(3, int(sisi * 0.02)))
            return cembung(rapi if rapi.sum() > 0.5 * isi.sum() else isi)
    return ndi.binary_dilation(g, iterations=2)

def haluskan(s):
    """Membersihkan tonjolan kecil, meratakan bentuk menjadi cembung, dan meluruskan tepinya."""
    sisi = max(s.shape)
    buka = ndi.binary_opening(s, structure=np.ones((3, 3)), iterations=max(3, int(sisi * 0.02)))
    pakai = buka if buka.sum() > 0.5 * s.sum() else s
    ys, xs = np.where(pakai)
    if len(ys) < 5:
        return s
    try:
        titik = np.stack([xs, ys], axis=1).astype('float64')
        P = [tuple(titik[i]) for i in ConvexHull(titik).vertices]
        luas = ConvexHull(titik).volume
        def segitiga(a, b, c):
            return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
        while len(P) > 3:
            n = len(P)
            kecil = [segitiga(P[i - 1], P[i], P[(i + 1) % n]) for i in range(n)]
            i = int(np.argmin(kecil))
            if kecil[i] > HALUS_SUDUT * luas:
                break
            P.pop(i)
        kanvas = Image.new('L', (s.shape[1], s.shape[0]), 0)
        ImageDraw.Draw(kanvas).polygon([(float(x), float(y)) for x, y in P], fill=1, outline=1)
        return np.array(kanvas).astype(bool)
    except Exception:
        return s

def normalisasi(s):
    """Mengubah siluet ke format 64x64px"""
    ys, xs = np.where(s)
    if len(ys) == 0:
        return None
    potongan = s[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype('float32') * 255
    h, w = potongan.shape
    sisi = max(h, w)
    kotak = np.zeros((sisi, sisi), dtype='float32')
    kotak[(sisi - h) // 2:(sisi - h) // 2 + h, (sisi - w) // 2:(sisi - w) // 2 + w] = potongan
    dalam = UKURAN_MODEL - 2 * MARGIN_HASIL
    kecil = Image.fromarray(kotak.clip(0, 255).astype('uint8')).resize((dalam, dalam), Image.LANCZOS)
    kanvas = np.zeros((UKURAN_MODEL, UKURAN_MODEL), dtype='float32')
    kanvas[MARGIN_HASIL:MARGIN_HASIL + dalam, MARGIN_HASIL:MARGIN_HASIL + dalam] = np.array(kecil)
    return (kanvas / 255).clip(0, 1).astype('float32')


def urutkan(kandidat):
    tinggi_med = np.median([k['tinggi'] for k in kandidat])
    kandidat = sorted(kandidat, key=lambda k: k['cy'])
    baris = []
    for k in kandidat:
        if baris and abs(k['cy'] - np.mean([x['cy'] for x in baris[-1]])) < 0.6 * tinggi_med:
            baris[-1].append(k)
        else:
            baris.append([k])
    return [sorted(b, key=lambda k: k['cx']) for b in baris]


def ciri_bentuk(s):
    """Menilai ciri objek untuk menyaring tangan dan benda lainnya berdasarkan rasio panjang/lebar."""
    ys, xs = np.where(s)
    if len(ys) < 10:
        return 0.0, 99.0
    sisi = max(s.shape)
    utuh = ndi.binary_opening(s, structure=np.ones((3, 3)), iterations=max(2, int(sisi * 0.02)))
    pakai = utuh if utuh.sum() > 0.5 * s.sum() else s
    py, px = np.where(pakai)
    try:
        luas_bungkus = ConvexHull(np.stack([px, py], axis=1).astype('float64')).volume
        kepadatan = len(py) / max(luas_bungkus, 1.0)
    except Exception:
        kepadatan = 0.0
    titik = np.stack([xs, ys], axis=1).astype('float64')
    ev = np.sort(np.linalg.eigvalsh(np.cov((titik - titik.mean(axis=0)).T)))
    rasio = float(np.sqrt(ev[1] / max(ev[0], 1e-9)))
    return float(kepadatan), rasio


def buat_negatif(jumlah, benih=0):
    """Menghasilkan siluet buatan 64x64px untuk kelas negatif untuk melatih model agar tidak 
    salah menebak bentuk geometri, lalu menormalisasikannya agar seragam dengan siluet foto."""
    rng = np.random.default_rng(benih)
    K = 400
    hasil, percobaan = [], 0
    while len(hasil) < jumlah and percobaan < jumlah * 5:
        jenis = percobaan % 7
        percobaan += 1
        kanvas = Image.new('L', (K, K), 0)
        d = ImageDraw.Draw(kanvas)
        c = K / 2
        R = rng.uniform(80, 170)
        if jenis == 0:
            n = int(rng.integers(5, 9))
            t0 = rng.uniform(0, 2 * np.pi)
            titik = [(c + R * rng.uniform(0.93, 1.0) * np.cos(t0 + 2 * np.pi * i / n),
                      c + R * rng.uniform(0.93, 1.0) * np.sin(t0 + 2 * np.pi * i / n)) for i in range(n)]
            d.polygon(titik, fill=255)
        elif jenis == 1:
            n = int(rng.integers(4, 9))
            t0 = rng.uniform(0, 2 * np.pi)
            r2 = R * rng.uniform(0.35, 0.7)
            titik = []
            for i in range(2 * n):
                rr = R if i % 2 == 0 else r2
                titik.append((c + rr * np.cos(t0 + np.pi * i / n), c + rr * np.sin(t0 + np.pi * i / n)))
            d.polygon(titik, fill=255)
        elif jenis == 2:
            b = R * rng.uniform(0.3, 0.9)
            d.ellipse([c - R, c - b, c + R, c + b], fill=255)
            kanvas = kanvas.rotate(rng.uniform(0, 180), resample=Image.BICUBIC)
        elif jenis == 3:
            lebar = rng.uniform(R / 14, R / 7)
            d.rounded_rectangle([c - R, c - lebar, c + R, c + lebar], radius=int(lebar * rng.uniform(0.2, 1.0)), fill=255)
            kanvas = kanvas.rotate(rng.uniform(0, 180), resample=Image.BICUBIC)
        elif jenis == 4:
            yy, xx = np.mgrid[0:K, 0:K]
            noise = ndi.gaussian_filter(rng.normal(0, 1, (K, K)), rng.uniform(18, 40))
            noise = noise / (np.abs(noise).max() + 1e-9) - ((xx - c) ** 2 + (yy - c) ** 2) / (R * 1.6) ** 2
            m = noise > np.quantile(noise, rng.uniform(0.55, 0.8))
            lab, n_l = ndi.label(m)
            if n_l == 0:
                continue
            m = lab == (1 + int(np.argmax(np.bincount(lab.ravel())[1:])))
            kanvas = Image.fromarray(ndi.binary_fill_holes(m).astype('uint8') * 255)
        elif jenis == 5:
            a, b = R * rng.uniform(0.9, 1.0), R * rng.uniform(0.18, 0.4)
            d.rectangle([c - a, c - b, c + a, c + b], fill=255)
            d.rectangle([c - b, c - a, c + b, c + a], fill=255)
            kanvas = kanvas.rotate(rng.uniform(0, 90), resample=Image.BICUBIC)
        else:
            d.ellipse([c - R * 0.5, c - R * 0.5, c + R * 0.5, c + R * 0.5], fill=255)
            n = int(rng.integers(3, 7))
            t0 = rng.uniform(0, 2 * np.pi)
            lebar = int(R * rng.uniform(0.18, 0.3))
            for i in range(n):
                t = t0 + rng.uniform(-0.2, 0.2) + 2 * np.pi * i / n * rng.uniform(0.35, 0.6)
                pj = R * rng.uniform(0.9, 1.3)
                d.line([(c, c), (c + pj * np.cos(t), c + pj * np.sin(t))], fill=255, width=lebar)
                d.ellipse([c + pj * np.cos(t) - lebar / 2, c + pj * np.sin(t) - lebar / 2,
                           c + pj * np.cos(t) + lebar / 2, c + pj * np.sin(t) + lebar / 2], fill=255)
        arr = normalisasi(np.array(kanvas) > 127)
        if arr is not None:
            hasil.append(arr)
    return np.array(hasil, dtype='float32')


def _siapkan(foto, jendela, faktor=0.5, ketat=True):
    # 1. Ambil foto dan perkecil kalau sangat besar
    img = Image.fromarray(foto).convert('RGB')
    img.thumbnail((800, 800))
    asli = np.array(img)
    arr = asli.astype('float32')
    h0, w0 = arr.shape[:2]
    sisi_foto = min(h0, w0)
    peta = asli.copy()
    kertas = warna_kertas(arr)

    # 2. Haluskan bintik/noise sedikit, lalu cari garis
    arr = ndi.gaussian_filter(arr, sigma=(0.7, 0.7, 0))
    sinyal = sinyal_goresan(arr, sisi_foto, jendela, tinta_gelap(arr, kertas))
    if sinyal.max() < 15:
        return [], peta, 'Tidak ada bentuk yang terdeteksi di foto.'
    arr = sinyal / sinyal.max() * 255

    # 3. Tentukan mana yang dianggap garis, lalu buang bintik
    batas = max(otsu(arr) * faktor, 30)
    mask = arr > batas
    label_m, n_m = ndi.label(mask, structure=np.ones((3, 3)))
    if n_m > 0:
        nomor = np.arange(1, n_m + 1)
        luas_potongan = np.array(ndi.sum(mask.astype('float32'), label_m, index=nomor))
        luas_min = max(6, int(sisi_foto ** 2 * BUANG_BINTIK))
        mask = np.isin(label_m, nomor[luas_potongan >= luas_min])
    if not mask.any():
        return [], peta, 'Tidak ada bentuk yang terdeteksi (hanya bintik halus).'

    # 4. Telusuri garis dan sambungkan potongan sampai membentuk garis tepi tertutup
    label, objek = kumpulkan_objek(mask, sisi_foto)
    halus = ndi.gaussian_filter(asli.astype('float32'), sigma=(1.2, 1.2, 0))
    gy, gx = np.gradient(halus, axis=(0, 1))
    ketajaman = np.sqrt((gx ** 2 + gy ** 2).mean(axis=2))

    # 5. Periksa tiap objek
    margin = max(2, int(sisi_foto * MARGIN_TEPI))
    kandidat = []
    ditolak_latar = 0
    for i, sl in objek:
        g = mask[sl] & (label[sl] == i)
        luas = int(g.sum())
        if luas < 30:
            continue
        if (sl[0].start < margin or sl[1].start < margin
                or sl[0].stop > h0 - margin or sl[1].stop > w0 - margin):
            continue
        ys, xs = np.where(g)
        tinggi = ys.max() - ys.min() + 1
        lebar = xs.max() - xs.min() + 1
        besar = max(tinggi, lebar) / sisi_foto
        if besar < UKURAN_MIN or besar > UKURAN_MAX:
            continue

        cincin_tepi = ndi.binary_dilation(g, iterations=2) & ~ndi.binary_erosion(g, iterations=2)
        if np.percentile(ketajaman[sl][cincin_tepi], 85) < TAJAM_MIN:
            ditolak_latar += 1
            continue

        if not latar_mirip_kertas(asli, label, i, mask, sl, kertas, sisi_foto):
            ditolak_latar += 1
            continue
        kandidat.append(dict(
            sl=sl, g=g, luas=luas, tinggi=tinggi, lebar=lebar,
            ukuran=max(tinggi, lebar),
            cy=(sl[0].start + sl[0].stop) / 2,
            cx=(sl[1].start + sl[1].stop) / 2,
        ))

    if not kandidat:
        if ditolak_latar:
            return [], peta, 'Ada goresan di foto, tapi bukan di atas kertas yang sama.'
        if not objek:
            return [], peta, 'Ada goresan di foto, tapi garisnya tidak membentuk geometri yang tertutup.'
        return [], peta, 'Ada objek di foto, tapi ukurannya tidak cocok sebagai geometri.'

    # 5b. Ubah garis tepi menjadi siluet, lalu singkirkan objek lainnya
    for k in kandidat:
        k['s'] = siluet(k['g'])
    if ketat:
        lolos = []
        for k in kandidat:
            kepadatan, rasio = ciri_bentuk(k['s'])
            if kepadatan >= KEPADATAN_MIN and rasio <= RASIO_MAX:
                lolos.append(k)
        kandidat = lolos
        if not kandidat:
            return [], peta, 'Ada objek di foto, tapi bentuknya tidak mirip geometri yang ingin di deteksi.'

    # 6. Buang pengotor yang jauh lebih kecil dari bentuk terbesar
    terbesar = max(k['ukuran'] for k in kandidat)
    kandidat = [k for k in kandidat if k['ukuran'] >= 0.35 * terbesar]
    luas_terbesar = max(k['luas'] for k in kandidat)
    kandidat = [k for k in kandidat if k['luas'] >= LUAS_MIN_REL * luas_terbesar]
    kandidat = sorted(kandidat, key=lambda k: k['luas'], reverse=True)[:MAKS_BENTUK]

    # 7. Urutkan, lalu isi tiap garis tepi jadi siluet dan ubah jadi gambar 64x64px
    baris_hasil = []
    for baris in urutkan(kandidat):
        item = []
        for k in baris:
            s = haluskan(k['s'])
            k['arr'] = normalisasi(s)
            p = max(12, int(max(k['g'].shape) * 0.12))
            k['sil'] = s[p:-p, p:-p]
            if k['arr'] is not None:
                item.append(k)
        if item:
            baris_hasil.append(item)
    return baris_hasil, peta, ''


def siapkan_bentuk(foto, ketat=True):
    pesan_awal = None
    for faktor in FAKTOR_BATAS:
        for jendela in (JENDELA_LATAR, JENDELA_BESAR, JENDELA_PENUH):
            baris_hasil, peta, pesan = _siapkan(foto, jendela, faktor, ketat)
            if baris_hasil:
                return baris_hasil, peta, pesan
            if pesan_awal is None:
                pesan_awal = (peta, pesan)
    return [], pesan_awal[0], pesan_awal[1]


def warnai_kontur(peta, k, warna, tebal=2):
    """Gambar garis tepi hasil penelusuran di atas foto."""
    sub = peta[k['sl']]
    sil = k['sil']
    h, w = min(sub.shape[0], sil.shape[0]), min(sub.shape[1], sil.shape[1])
    sil = sil[:h, :w]
    garis = sil & ~ndi.binary_erosion(sil, iterations=tebal)
    garis = ndi.binary_dilation(garis, iterations=1)
    sub[:h, :w][garis] = warna


def beri_bingkai(foto, persen=0.15):
    """Tambah frame di sekeliling gambar, agar bentuk yang dekat dengan tepi tidak dibuang oleh filter 'menempel tepi foto'"""
    warna = warna_kertas(foto.astype('float32')).astype(foto.dtype)
    t = max(10, int(min(foto.shape[:2]) * persen))
    hasil = np.empty((foto.shape[0] + 2 * t, foto.shape[1] + 2 * t, 3), dtype=foto.dtype)
    hasil[:] = warna
    hasil[t:t + foto.shape[0], t:t + foto.shape[1]] = foto
    return hasil


def array_ke_rgb(foto):
    """Gambar (array) diubah ke array RGB. Lalu, gambar PNG diubah jadi berwarna hitam di kertas putih."""
    foto = np.asarray(foto)
    if foto.ndim == 2:
        foto = np.stack([foto] * 3, axis=2)
    if foto.shape[2] == 4:
        alpha = foto[:, :, 3]
        if (alpha < 30).mean() > 0.05 and (alpha > 200).mean() > 0.005:
            kertas = np.full(alpha.shape + (3,), 255, dtype="uint8")
            kertas[alpha >= 128] = 20
            return kertas
    return foto[:, :, :3].astype("uint8")


def buka_rgb(path):
    img = ImageOps.exif_transpose(Image.open(path))
    return array_ke_rgb(np.array(img.convert("RGBA")))
