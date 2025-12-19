#!/bin/bash

# Mensaje de inicio
echo "Iniciando secuencia de entrenamiento en Linux..."

# 1. Ejecutar ESDR x4
echo "Ejecutando: train_esdr_x4.py"
python3 train_esdr_x4.py

# 2. Ejecutar ESDR x2 con Augmentation
echo "Ejecutando: train_esdr_x2 ssim_amp_aug.py"
python3 "train_esdr_x2 ssim_amp.py"

# 3. Ejecutar ESDR x2 SSIM AMP
echo "Ejecutando: train_esdr_x2 ssim_amp.py"
python3 "train_esdr_x2 ssim_amp_aug.py"

# 4. Ejecutar ESDR x2 con 32 bloques
echo "Ejecutando: train_esdr_x2 ssim_amp_aug_32blocks.py"
python3 "train_esdr_x2 ssim_amp_aug_32blocks.py"

echo "Proceso completado."