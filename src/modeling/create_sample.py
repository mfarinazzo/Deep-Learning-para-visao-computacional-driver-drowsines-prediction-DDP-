import argparse
import random
import shutil
import sys
from pathlib import Path

# --- LOCALIZAÇÃO DA RAIZ DO PROJETO ---
# O arquivo está em: PROJECT_ROOT/src/modeling/create_sample.py
# parents[0] = src/modeling
# parents[1] = src
# parents[2] = PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

def create_sample(src_rel: str, dst_rel: str, samples_per_class: int):
    # Converte caminhos relativos para absolutos baseados na raiz do projeto
    src_root = (PROJECT_ROOT / src_rel).resolve()
    dst_root = (PROJECT_ROOT / dst_rel).resolve()

    print(f"\n=== DEBUG DE CAMINHOS ===")
    print(f"Raiz do Projeto Detectada: {PROJECT_ROOT}")
    print(f"Lendo de (Source):         {src_root}")
    print(f"Escrevendo em (Dest):      {dst_root}")
    print("=========================\n")

    if not src_root.exists():
        print(f"ERRO: A pasta de origem não existe no caminho absoluto acima!")
        sys.exit(1)

    # Verifica se a origem tem conteúdo
    files_in_src = list(src_root.glob("**/*.jpg")) + list(src_root.glob("**/*.png"))
    if not files_in_src:
        print(f"ERRO: A pasta de origem existe, mas está VAZIA (0 imagens encontradas).")
        print("Rode o pipeline anterior: python src/run_pipeline.py ...")
        sys.exit(1)
    
    print(f"Encontradas {len(files_in_src)} imagens na origem. Iniciando amostragem...")

    if dst_root.exists():
        print(f"Removendo pasta antiga: {dst_root}")
        shutil.rmtree(dst_root)
    
    total_copied = 0
    
    for original_split, yolo_split in SPLIT_MAP.items():
        src_split = src_root / original_split
        dst_split = dst_root / yolo_split
        
        if not src_split.exists():
            print(f"[Aviso] Split não encontrado na origem: {src_split.name}")
            continue
            
        # Itera sobre as classes
        for class_dir in src_split.iterdir():
            if not class_dir.is_dir():
                continue
                
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            
            if not images:
                continue

            k = min(len(images), samples_per_class)
            selected = random.sample(images, k)
            
            target_dir = dst_split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for img in selected:
                shutil.copy2(img, target_dir / img.name)
                total_copied += 1
            
            # Feedback visual simples (um ponto por classe)
            print(f".", end="", flush=True)

    print(f"\n\nSucesso! {total_copied} imagens copiadas.")
    print(f"CONFIRA A PASTA AGORA EM: {dst_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Note que agora os defaults são strings relativas simples
    parser.add_argument("--src", type=str, default="data/standardized")
    parser.add_argument("--dst", type=str, default="data/sample")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    
    create_sample(args.src, args.dst, args.n)
