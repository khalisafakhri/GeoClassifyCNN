# -*- coding: utf-8 -*-
"""
KLASIFIKASI BENTUK: JAJAR GENJANG, PERSEGI PANJANG, PERSEGI (CNN)

1. Dataset        -> download otomatis dari Kaggle, lalu cek jumlah gambar
2. Contoh gambar  -> lihat isi dataset
3. Pengolah foto  -> memuat pengolah_bentuk.py
4. Data siluet    -> dataset diproses oleh pengolah yang sama, disimpan ke data_siluet.npz
5. Latih model    -> CNN, uji di data test, tabel kebingungan, simpan model
6. Program akhir  -> fungsi baca_bentuk(foto)
7. Coba           -> beri foto, keluar "Ini jajar genjang", dst.

"""

# %% 1. Dataset: download otomatis dan cek isinya

import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

FOLDER_PROYEK = r"C:\Users\fakhr\Downloads\proyek_bentuk"
FOLDER_DATASET = os.path.join(FOLDER_PROYEK, "dataset")
os.makedirs(FOLDER_PROYEK, exist_ok=True)


def cari_data_dir():
    for root, dirs, files in os.walk(FOLDER_PROYEK):
        if "train" in dirs and "val" in dirs and "test" in dirs:
            return root
    return None


DATA_DIR = cari_data_dir()
if DATA_DIR is None:
    import kagglehub
    kagglehub.dataset_download("reevald/geometric-shapes-mathematics", output_dir=FOLDER_DATASET)
    DATA_DIR = cari_data_dir()
print("DATA_DIR:", DATA_DIR)

KELAS = ["parallelogram", "rectangle", "square"]

for split in ["train", "val", "test"]:
    for kelas in KELAS:
        folder = os.path.join(DATA_DIR, split, kelas)
        print(split, kelas, "->", len(os.listdir(folder)), "gambar")


# %% 2. Lihat contoh gambar

plt.figure(figsize=(9, 9))
n = 1
for kelas in KELAS:
    folder = os.path.join(DATA_DIR, "train", kelas)
    for f in sorted(os.listdir(folder))[:3]:
        img = Image.open(os.path.join(folder, f))
        print(kelas, f, "| ukuran:", img.size, "| mode:", img.mode)
        plt.subplot(3, 3, n)
        plt.imshow(img)
        plt.title(kelas)
        plt.axis("off")
        n += 1
plt.show()

# %% 3. Pengolah foto: dibaca dari file pengolah_bentuk.py

import sys
import importlib

if not os.path.exists(os.path.join(FOLDER_PROYEK, "pengolah_bentuk.py")):
    raise FileNotFoundError("Simpan dulu file pengolah_bentuk.py di folder ini: " + FOLDER_PROYEK)
if FOLDER_PROYEK not in sys.path:
    sys.path.insert(0, FOLDER_PROYEK)
import pengolah_bentuk
importlib.reload(pengolah_bentuk)
from pengolah_bentuk import (siapkan_bentuk, beri_bingkai, buka_rgb, warnai_kontur, buat_negatif,
                             UKURAN_MODEL, MARGIN_HASIL, VERSI_PENGOLAH)
print("Pengolah foto dimuat dari:", pengolah_bentuk.__file__)

# %% 4. Menyiapkan data latih

import time

# "rhombus" (belah ketupat) dari dataset yang sama ditambahkan untuk dijadikan kelasnya sendiri,
# supaya belah ketupat tidak jadi persegi atau jajar genjang.

PAKAI_BUKAN = True
KELAS_BUKAN = ["circle", "kite", "trapezoid", "triangle", "rhombus"]

FILE_CACHE = os.path.join(FOLDER_PROYEK, "data_siluet.npz")
KELAS_UTAMA = KELAS
NAMA_KELAS = KELAS_UTAMA + (["bukan"] if PAKAI_BUKAN else [])


def siluet_dataset(path):
    foto = buka_rgb(path)
    baris, _, _ = siapkan_bentuk(beri_bingkai(foto), ketat=False)
    hasil = [k for b in baris for k in b]
    if not hasil:
        return None
    return max(hasil, key=lambda k: k["ukuran"])["arr"]


def proses_folder(folder, batas):
    daftar = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    daftar = daftar[:batas]
    X, gagal = [], 0
    for f in daftar:
        arr = siluet_dataset(os.path.join(folder, f))
        if arr is None:
            gagal += 1
        else:
            X.append(arr)
    return X, gagal, len(daftar)


def bangun_split(split):
    Xs, ys = [], []
    for label, kelas in enumerate(KELAS_UTAMA):
        X, gagal, total = proses_folder(os.path.join(DATA_DIR, split, kelas), 10 ** 9)
        Xs += X
        ys += [label] * len(X)
        print(f"  {split:5s} {kelas:14s} berhasil {len(X)}/{total}  (gagal {gagal})")
    if PAKAI_BUKAN:
        besar_kelas = len(os.listdir(os.path.join(DATA_DIR, split, KELAS[0])))
        n_buatan = besar_kelas // 4
        per_kategori = (besar_kelas - n_buatan) // len(KELAS_BUKAN)
        berhasil, gagal_b, total_b = 0, 0, 0
        for kelas in KELAS_BUKAN:
            X, gagal, total = proses_folder(os.path.join(DATA_DIR, split, kelas), per_kategori)
            Xs += X
            ys += [len(KELAS_UTAMA)] * len(X)
            berhasil += len(X)
            gagal_b += gagal
            total_b += total
        buatan = buat_negatif(n_buatan, {"train": 1, "val": 2, "test": 3}[split])
        Xs += list(buatan)
        ys += [len(KELAS_UTAMA)] * len(buatan)
        print(f"  {split:5s} {'bukan':14s} berhasil {berhasil}/{total_b}  (gagal {gagal_b})  + {len(buatan)} bentuk buatan")
    return np.array(Xs, dtype="float32"), np.array(ys)


cache_cocok = False
if os.path.exists(FILE_CACHE):
    isi = np.load(FILE_CACHE, allow_pickle=False)
    cache_cocok = (list(isi["nama_kelas"]) == NAMA_KELAS and int(isi["margin"]) == MARGIN_HASIL
                   and "versi" in isi.files and int(isi["versi"]) == VERSI_PENGOLAH)

if cache_cocok:
    print("Memakai data yang sudah diproses sebelumnya:", FILE_CACHE)
    X_train, y_train = isi["X_train"], isi["y_train"]
    X_val, y_val = isi["X_val"], isi["y_val"]
    X_test, y_test = isi["X_test"], isi["y_test"]
else:
    print("Memproses dataset")
    t0 = time.time()
    X_train, y_train = bangun_split("train")
    X_val, y_val = bangun_split("val")
    X_test, y_test = bangun_split("test")
    np.savez_compressed(FILE_CACHE, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test,
                        nama_kelas=np.array(NAMA_KELAS), margin=np.array(MARGIN_HASIL),
                        versi=np.array(VERSI_PENGOLAH))
    print("Selesai dalam %.0f detik. Tersimpan:" % (time.time() - t0), FILE_CACHE)

print("Data latih:", X_train.shape, "| per kelas:", np.bincount(y_train, minlength=len(NAMA_KELAS)))
print("Data val  :", X_val.shape, "| per kelas:", np.bincount(y_val, minlength=len(NAMA_KELAS)))
print("Data test :", X_test.shape, "| per kelas:", np.bincount(y_test, minlength=len(NAMA_KELAS)))

X_latih = X_train
y_latih = y_train

# Lihat contoh hasil yang di process
plt.figure(figsize=(12, 1.8 * len(NAMA_KELAS)))
for baris_k, nama in enumerate(NAMA_KELAS):
    idx = np.where(y_train == baris_k)[0][:8]
    for kolom, i in enumerate(idx):
        plt.subplot(len(NAMA_KELAS), 8, baris_k * 8 + kolom + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.axis("off")
        if kolom == 0:
            plt.title(nama, fontsize=9, loc="left")
plt.show()

# %% 5. Latih model CNN, uji, dan simpan

import json
from tensorflow import keras

EPOCHS = 25
JUMLAH_KELAS = len(NAMA_KELAS)

# Augmentasi
AUGMENTASI_PUTAR = True

augmentasi = [keras.layers.RandomTranslation(0.05, 0.05, fill_mode='constant', fill_value=0.0)]
if AUGMENTASI_PUTAR:
    augmentasi = [keras.layers.RandomFlip("horizontal_and_vertical"),
                  keras.layers.RandomRotation(0.5, fill_mode='constant', fill_value=0.0)] + augmentasi

model = keras.Sequential([
    keras.Input(shape=(UKURAN_MODEL, UKURAN_MODEL)),
    keras.layers.Reshape((UKURAN_MODEL, UKURAN_MODEL, 1)),
] + augmentasi + [
    keras.layers.Conv2D(32, 3, activation='relu'),       # mencari 32 pola kecil
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),       # mencari 64 pola yang lebih rumit
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation='relu'),      # mencari 128 pola yang lebih rumit lagi
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),                              # diratakan jadi satu baris
    keras.layers.Dropout(0.3),                           # mematikan 30% neuron acak
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(JUMLAH_KELAS, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

berhenti_dini = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_latih, y_latih,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=32,
                    callbacks=[berhenti_dini])


plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Akurasi")
plt.xlabel("epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

# Melakukan uji
loss, akurasi = model.evaluate(X_test, y_test, verbose=0)
print("Akurasi di data test:", round(akurasi * 100, 2), "%")

y_tebak = np.argmax(model.predict(X_test, verbose=0), axis=1)
tabel = np.zeros((JUMLAH_KELAS, JUMLAH_KELAS), dtype=int)
for benar, tebak in zip(y_test, y_tebak):
    tabel[benar, tebak] += 1
print("\nTabel kebingungan (baris = jawaban benar, kolom = tebakan model)")
print("urutan:", NAMA_KELAS)
print(tabel)
for i, nama in enumerate(NAMA_KELAS):
    print(nama, ":", round(100 * tabel[i, i] / max(tabel[i].sum(), 1), 1), "% benar")

# Contoh tebakan yang salah
salah = np.where(y_tebak != y_test)[0][:8]
if len(salah) > 0:
    plt.figure(figsize=(12, 2.5))
    for kolom, i in enumerate(salah):
        plt.subplot(1, 8, kolom + 1)
        plt.imshow(X_test[i], cmap="gray")
        plt.title(f"asli: {NAMA_KELAS[y_test[i]]}\ntebak: {NAMA_KELAS[y_tebak[i]]}", fontsize=7)
        plt.axis("off")
    plt.show()

# Menyimpan model
model.save(os.path.join(FOLDER_PROYEK, "Model_bentuk_siluet.keras"))
with open(os.path.join(FOLDER_PROYEK, "nama_kelas.json"), "w", encoding="utf-8") as f:
    json.dump(NAMA_KELAS, f)
print("\nTersimpan: Model_bentuk_siluet.keras dan nama_kelas.json di", FOLDER_PROYEK)

# %% 6. Program akhir

import json
from tensorflow import keras

model_akhir = keras.models.load_model(os.path.join(FOLDER_PROYEK, "Model_bentuk_siluet.keras"))
with open(os.path.join(FOLDER_PROYEK, "nama_kelas.json"), encoding="utf-8") as f:
    NAMA_MODEL = json.load(f)

NAMA_INDO = {
    "parallelogram": "jajar genjang",
    "rectangle": "persegi panjang",
    "square": "persegi",
    "bukan": "Bukan jajar genjang, persegi panjang, atau persegi"
}
BATAS_YAKIN = 0.75

def baca_bentuk(path_foto, tampilkan=True):
    foto = buka_rgb(path_foto)
    baris, peta, pesan = siapkan_bentuk(foto)
    hasil = [k for b in baris for k in b]
    if not hasil:
        baris2, peta2, pesan2 = siapkan_bentuk(beri_bingkai(foto))
        hasil2 = [k for b in baris2 for k in b]
        if hasil2:
            hasil, peta, pesan = hasil2, peta2, pesan2
    if not hasil:
        print(pesan)
        return []

    X = np.stack([k["arr"] for k in hasil]).astype("float32")
    prob = model_akhir.predict(X, verbose=0)

    keluaran = []
    for n, (k, p) in enumerate(zip(hasil, prob), start=1):
        idx = int(np.argmax(p))
        nama = NAMA_MODEL[idx]
        yakin = float(p[idx])
        if nama == "bukan":
            teks = "Ini" + NAMA_INDO["bukan"]
            yakin_ok = True
        elif yakin < BATAS_YAKIN:
            teks = f"Tidak yakin, mungkin ini adalah bentuk {NAMA_INDO.get(nama, nama)}"
            yakin_ok = False
        else:
            teks = f"Ini adalah bentuk {NAMA_INDO.get(nama, nama)}"
            yakin_ok = True
        print(f"Bentuk {n}: {teks} (keyakinan {yakin * 100:.1f}%)")
        keluaran.append(dict(nomor=n, nama=nama, teks=teks, yakin=yakin, sl=k["sl"], sil=k["sil"], ok=yakin_ok))

    if tampilkan:
        gambar = peta.copy()
        for h in keluaran:
            h["warna"] = (0, 200, 0) if h["ok"] and h["nama"] != "bukan" else (255, 140, 0)
            warnai_kontur(gambar, h, h["warna"])
        plt.figure(figsize=(9, 6))
        ax = plt.gca()
        ax.imshow(gambar)
        for h in keluaran:
            ax.text(h["sl"][1].start, h["sl"][0].start - 6, f"{h['nomor']}. {h['teks']}",
                    color=tuple(c / 255 for c in h["warna"]), fontsize=11, backgroundcolor="black")
        ax.axis("off")
        plt.show()
    return keluaran


# %% 7. Gambar confusion matrix

import os
import numpy as np
import matplotlib.pyplot as plt

# Nama yang tampil di gambar
NAMA_TAMPIL = {
    "parallelogram": "Jajar genjang",
    "rectangle": "Persegi panjang",
    "square": "Persegi",
    "bukan": "Bukan",
}
nama_tampil = [NAMA_TAMPIL.get(n, n) for n in NAMA_KELAS]
jumlah = len(nama_tampil)

# 1. Model menebak semua data uji
y_tebak = np.argmax(model.predict(X_test, verbose=0), axis=1)

# 2. Hitung tabel: baris = jawaban benar, kolom = tebakan model
tabel = np.zeros((jumlah, jumlah), dtype=int)
for benar, tebak in zip(y_test, y_tebak):
    tabel[benar, tebak] += 1

# Persen per baris (berapa persen dari tiap jawaban benar yang jatuh ke tiap tebakan)
persen = tabel / np.maximum(tabel.sum(axis=1, keepdims=True), 1)

# 3. Gambar tabelnya
fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=200)
ax.imshow(persen, cmap="Blues", vmin=0, vmax=1)

for i in range(jumlah):
    for j in range(jumlah):
        warna = "white" if persen[i, j] > 0.55 else "#1a1a1a"
        ax.text(j, i - 0.08, f"{tabel[i, j]}", ha="center", va="center",
                fontsize=17, fontweight="bold", color=warna)
        ax.text(j, i + 0.24, f"{100 * persen[i, j]:.1f}%", ha="center", va="center",
                fontsize=10.5, color=warna)

ax.set_xticks(range(jumlah))
ax.set_yticks(range(jumlah))
ax.set_xticklabels(nama_tampil, fontsize=11)
ax.set_yticklabels(nama_tampil, fontsize=11)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.set_xlabel("Tebakan model", fontsize=12, labelpad=10)
ax.set_ylabel("Jawaban benar", fontsize=12)
for sisi in ax.spines.values():
    sisi.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, jumlah, 1), minor=True)
ax.set_yticks(np.arange(-0.5, jumlah, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="minor", length=0)
plt.tight_layout()

# 4. Simpan sebagai gambar, lalu tampilkan
path_gambar = os.path.join(FOLDER_PROYEK, "matriks_kebingungan.png")
plt.savefig(path_gambar, facecolor="white")
plt.show()
print("Gambar tersimpan di:", path_gambar)

# 5. Ringkasan angka
print("\nAkurasi data uji:", round(100 * np.trace(tabel) / tabel.sum(), 2), "%")
print("kelas | benar (recall) | presisi")
for i, nama in enumerate(nama_tampil):
    recall = 100 * tabel[i, i] / max(tabel[i].sum(), 1)
    presisi = 100 * tabel[:, i].sum() and 100 * tabel[i, i] / max(tabel[:, i].sum(), 1)
    print(f"{nama} | {recall:.1f}% | {presisi:.1f}%")
    
# %% 8. Grafik

import os
import numpy as np
import matplotlib.pyplot as plt

# 1. Ambil catatan latihan dari sel 5
riwayat = history.history
jumlah_epoch = len(riwayat["loss"])
epoch = np.arange(1, jumlah_epoch + 1)

terbaik = int(np.argmin(riwayat["val_loss"])) + 1

# 2. Gambar dua grafik
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)

for ax, kunci, judul in [(ax1, "accuracy", "Akurasi"), (ax2, "loss", "Loss")]:
    ax.plot(epoch, riwayat[kunci], color="#1f5fa8", linewidth=2, marker="o", markersize=3.5, label="Latih")
    ax.plot(epoch, riwayat["val_" + kunci], color="#e8821e", linewidth=2, marker="o", markersize=3.5, label="Validasi")
    ax.axvline(terbaik, color="gray", linestyle="--", linewidth=1)
    ax.set_title(judul, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_xticks(epoch if jumlah_epoch <= 15 else np.arange(1, jumlah_epoch + 1, 2))
    ax.grid(alpha=0.25)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

ax1.text(terbaik, ax1.get_ylim()[0], f" epoch terbaik ({terbaik})", color="gray", fontsize=9, va="bottom")
plt.tight_layout()

# 3. Simpan sebagai gambar, lalu tampilkan
path_gambar = os.path.join(FOLDER_PROYEK, "grafik_latihan.png")
plt.savefig(path_gambar, facecolor="white")
plt.show()
print("Gambar tersimpan di:", path_gambar)

# 4. Ringkasan angka untuk slide
print("\nEpoch yang dijalankan:", jumlah_epoch, "dari maksimal", EPOCHS)
if jumlah_epoch < EPOCHS:
    print(f"Latihan berhenti di epoch {jumlah_epoch} karena val_loss tidak membaik 5 epoch (EarlyStopping).")
else:
    print("Latihan berjalan sampai batas epoch maksimal.")
print(f"Bobot terbaik dari epoch {terbaik}: "
      f"val_accuracy = {riwayat['val_accuracy'][terbaik - 1]:.4f}, val_loss = {riwayat['val_loss'][terbaik - 1]:.4f}")
print(f"Akurasi latih di epoch itu: {riwayat['accuracy'][terbaik - 1]:.4f}")