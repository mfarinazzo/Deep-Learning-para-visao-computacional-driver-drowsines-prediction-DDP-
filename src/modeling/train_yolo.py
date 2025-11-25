import matplotlib
# --- VACINA PARA MAC OS ---
matplotlib.use('Agg') 
# --------------------------

import ssl
import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

# --- SSL FIX ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# ---------------

def get_device(requested_device=None):
    if requested_device:
        return requested_device
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def train(data_dir: str, epochs: int, imgsz: int, batch: int, project_name: str, device_arg: str):
    data_path = Path(data_dir)
    
    device = get_device(device_arg)
    
    print(f"Carregando dados de: {data_dir}")
    print(f"--- MODO HEADLESS (SEM GUI) ---")
    print(f"Dispositivo: {device}")
    
    model = YOLO('yolov8n-cls.pt') 

    try:
        results = model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=project_name,
            name='yolo_driver_drowsiness',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            
            # CONFIGURAÇÕES ORIGINAIS (SEM EXTRAS)
            device=device,
            workers=0,      # Força 0 para não crashar no Mac
            amp=False,      
            plots=False,    
            val=True,
            cache=True
        )
        print("Treinamento concluído com sucesso.")
    except Exception as e:
        print(f"\n ERRO CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=str, default="outputs/training_logs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    abs_data_path = str(Path(args.data).resolve())
    train(abs_data_path, args.epochs, args.imgsz, args.batch, args.project, args.device)