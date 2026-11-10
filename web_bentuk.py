# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:31:53 2026

@author: fakhr
"""

# -*- coding: utf-8 -*-
"""
WEBSITE KLASIFIKASI BENTUK

Alur kerja:
1. Pengguna upload atau foto kertas berisi bentuk gambar tangan.
2. Kode mencari garis gambar di foto.
3. Garisnya ditelusuri.
4. Model menebak tiap bentuk.
5. Hasilnya ditampilkan di web.

"""

import os
import sys
import json
import importlib
import traceback
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras

# Lokasi folder project
FOLDER_PROYEK = r"C:\Users\fakhr\Downloads\proyek_bentuk"

# Memuat pengolah foto dari pengolah_bentuk.py
if not os.path.exists(os.path.join(FOLDER_PROYEK, "pengolah_bentuk.py")):
    raise FileNotFoundError("Simpan dulu file pengolah_bentuk.py di folder ini: " + FOLDER_PROYEK)
if FOLDER_PROYEK not in sys.path:
    sys.path.insert(0, FOLDER_PROYEK)
import pengolah_bentuk
importlib.reload(pengolah_bentuk)
from pengolah_bentuk import siapkan_bentuk, beri_bingkai, array_ke_rgb, warnai_kontur, UKURAN_MODEL

# Memuat model dan daftar nama kelasnya
model = keras.models.load_model(os.path.join(FOLDER_PROYEK, "Model_bentuk_siluet.keras"))
with open(os.path.join(FOLDER_PROYEK, "nama_kelas.json"), encoding="utf-8") as f:
    NAMA_MODEL = json.load(f)

NAMA_INDO = {
    "parallelogram": "jajar genjang",
    "rectangle": "persegi panjang",
    "square": "persegi",
    "bukan": "bukan jajar genjang, persegi panjang, maupun persegi",
}

# Bentuk hanya dianggap terbaca kalau keyakinan model di atas 75%
BATAS_YAKIN = 0.75

try:
    FONT = ImageFont.load_default(size=30)
except TypeError:
    FONT = ImageFont.load_default()


def susun_gambar(tiles):
    """Menyusun banyak gambar kecil menjadi satu gambar."""
    per_baris = 6
    kolom = min(per_baris, len(tiles))
    kosong = np.pad(np.zeros_like(tiles[0]), 4, constant_values=80)
    tiles = [np.pad(t, 4, constant_values=80) for t in tiles]
    baris_img = []
    for s in range(0, len(tiles), per_baris):
        row = tiles[s:s + per_baris]
        row = row + [kosong] * (kolom - len(row))
        baris_img.append(np.concatenate(row, axis=1))
    return np.concatenate(baris_img, axis=0)


# Fungsi yang dipanggil Web

def tebak(foto):
    if foto is None:
        return '', None, None

    try:
        foto = array_ke_rgb(foto)
        baris, peta, catatan = siapkan_bentuk(foto)
        semua = [k for b in baris for k in b]
        if not semua:
            baris2, peta2, catatan2 = siapkan_bentuk(beri_bingkai(foto))
            semua2 = [k for b in baris2 for k in b]
            if semua2:
                baris, peta, catatan, semua = baris2, peta2, catatan2, semua2

        if not semua:
            return f'##Tidak ada bentuk yang diprediksi\n{catatan}', None, peta

        batch = np.stack([k['arr'] for k in semua]).reshape(-1, UKURAN_MODEL, UKURAN_MODEL)
        hasil = model.predict(batch, verbose=0)

        for k, h in zip(semua, hasil):
            idx = int(np.argmax(h))
            k['nama'] = NAMA_MODEL[idx]
            k['yakin'] = float(h[idx])
            if k['nama'] == 'bukan':
                k['status'] = 'bukan'
            elif k['yakin'] < BATAS_YAKIN:
                k['status'] = 'ragu'
            else:
                k['status'] = 'bentuk'

        warna = {'bentuk': (0, 200, 0), 'ragu': (255, 200, 0), 'bukan': (255, 0, 0)}

        rincian = []
        for no, k in enumerate(semua, start=1):
            nama = NAMA_INDO[k['nama']]
            if k['status'] == 'bentuk':
                rincian.append(f"{no}. **{nama}** ({k['yakin'] * 100:.0f}%)")
            elif k['status'] == 'ragu':
                rincian.append(f"{no}. **Model ragu untuk menebaknya** (Tebakan terdekat {nama}, sebesar {k['yakin'] * 100:.0f}%)")
            else:
                rincian.append(f"{no}. Bukan jajar genjang, persegi panjang, atau persegi ({k['yakin'] * 100:.0f}%)")

        # Gambar garis tepi hasil penelusuran dan manambahkan nomor urut di atas foto
        peta = peta.copy()
        for k in semua:
            warnai_kontur(peta, k, warna[k['status']])
        im = Image.fromarray(peta)
        d = ImageDraw.Draw(im)
        for no, k in enumerate(semua, start=1):
            d.text((k['sl'][1].start, max(k['sl'][0].start - 34, 0)), str(no),
                   fill=warna[k['status']], font=FONT)
        peta = np.array(im)

        # Gambar yang benar-benar dilihat model
        tiles = [np.array(Image.fromarray((k['arr'] * 255).astype('uint8')).resize((128, 128), Image.NEAREST))
                 for k in semua]
        dilihat = susun_gambar(tiles)

        # Menampilkan hasil, yaitu judul dan daftar bernomor
        md = '## Hasil\n\n' + '\n'.join(rincian)
        return md, dilihat, peta

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f'{type(e).__name__}: {e}')


# Interface Web

web = gr.Interface(
    fn=tebak,
    inputs=gr.Image(
        type='numpy',
        image_mode='RGBA',
        label='Upload foto bentuk jajar genjang/persegi panjang/persegi',
        sources=['upload', 'webcam', 'clipboard'],
        webcam_options=gr.WebcamOptions(mirror=False),
    ),
    outputs=[
        gr.Markdown(),
        gr.Image(label='Yang dilihat model'),
        gr.Image(label='Pengecekan (Warna hijau = dikenali, warna kuning = ragu, warna merah = bukan)'),
    ],
    title='Sistem Klasifikasi Bentuk berbasis Neural Network',
    description='Upload foto jajar genjang, persegi panjang, atau persegi, lalu model akan menebaknya!',
)

if __name__ == '__main__':
    web.launch(share=True)