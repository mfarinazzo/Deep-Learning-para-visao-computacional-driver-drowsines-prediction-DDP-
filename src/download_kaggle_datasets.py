from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any


# ALTERAÇÃO: ROOT agora aponta para a raiz do projeto (parents[1]), não para src
ROOT = Path(__file__).resolve().parents[1]


def retry_operation(func: Callable, *args, retries: int = 5, delay: float = 2.0, **kwargs) -> Any:
    """Executa uma função com tentativas (retries) e backoff exponencial."""
    last_exception = None
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            wait = delay * (2 ** i)
            print(f"\n[Aviso] Falha na tentativa {i+1}/{retries}. Erro: {e}")
            print(f"Aguardando {wait:.1f}s para tentar novamente...")
            time.sleep(wait)
    print("\n[Erro] Todas as tentativas falharam.")
    raise last_exception


def find_kaggle_json(config_dir_hint: Optional[Path] = None) -> Optional[Path]:
    candidates = []
    if config_dir_hint:
        candidates.append(config_dir_hint / "kaggle.json")
    # Kaggle default
    candidates.append(Path.home() / ".kaggle" / "kaggle.json")
    # Project-local
    candidates.append(ROOT / ".kaggle" / "kaggle.json")
    for p in candidates:
        if p.exists():
            return p
    return None


def ensure_kaggle_cli() -> str | None:
    try:
        subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return "kaggle"
    except Exception:
        pass
    try:
        subprocess.run([sys.executable, "-m", "kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"{sys.executable} -m kaggle"
    except Exception:
        return None


def set_kaggle_env(kaggle_json: Path) -> None:
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)


def run_kaggle_cmd(args: List[str], cwd: Optional[Path] = None, runner: Optional[str] = None) -> None:
    cmd: List[str]
    if runner and runner != "kaggle":
        parts = runner.split()
        cmd = [*parts, *args]
    else:
        cmd = ["kaggle", *args]
    
    def _execute():
        # Stream output in real time
        proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Kaggle command failed (exit {ret})")

    retry_operation(_execute)


def unzip_all_in_dir(zip_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for z in zip_dir.glob("*.zip"):
        print(f"Extraindo {z.name}...")
        try:
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(dest)
        except zipfile.BadZipFile:
            print(f"[Erro] Arquivo zip corrompido: {z}")


def download_datasets_cli(cfg: Dict, apply: bool, runner: Optional[str]) -> None:
    datasets = cfg.get("datasets", [])
    for ds in datasets:
        if not ds.get("enabled", False):
            continue
        ds_id = ds["id"]
        # Caminho relativo ao ROOT do projeto
        path = ROOT / ds.get("path", "raw")
        unzip = bool(ds.get("unzip", True))
        path.mkdir(parents=True, exist_ok=True)
        print(f"Dataset: {ds_id} -> {path}")
        if not apply:
            continue
        run_kaggle_cmd(["datasets", "download", "-d", ds_id, "-p", str(path), "--force"], runner=runner)
        if unzip:
            unzip_all_in_dir(path, path)


def download_competitions_cli(cfg: Dict, apply: bool, runner: Optional[str]) -> None:
    comps = cfg.get("competitions", [])
    for cp in comps:
        if not cp.get("enabled", False):
            continue
        cp_id = cp["id"]
        path = ROOT / cp.get("path", "raw")
        unzip = bool(cp.get("unzip", True))
        path.mkdir(parents=True, exist_ok=True)
        print(f"Competition: {cp_id} -> {path}")
        if not apply:
            continue
        run_kaggle_cmd(["competitions", "download", "-c", cp_id, "-p", str(path), "--force"], runner=runner)
        if unzip:
            unzip_all_in_dir(path, path)


def download_datasets_api(cfg: Dict, apply: bool) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("Kaggle API is not available; ensure 'kaggle' package is installed.") from e
    
    api = KaggleApi()
    api.authenticate()
    datasets = cfg.get("datasets", [])
    
    for ds in datasets:
        if not ds.get("enabled", False):
            continue
        ds_id = ds["id"]
        path = ROOT / ds.get("path", "raw")
        unzip = bool(ds.get("unzip", True))
        path.mkdir(parents=True, exist_ok=True)
        print(f"Dataset (API): {ds_id} -> {path}")
        if not apply:
            continue
        
        retry_operation(
            api.dataset_download_files, 
            ds_id, 
            path=str(path), 
            unzip=unzip, 
            quiet=False, 
            force=True
        )


def download_competitions_api(cfg: Dict, apply: bool) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("Kaggle API is not available; ensure 'kaggle' package is installed.") from e
    
    api = KaggleApi()
    api.authenticate()
    comps = cfg.get("competitions", [])
    for cp in comps:
        if not cp.get("enabled", False):
            continue
        cp_id = cp["id"]
        path = ROOT / cp.get("path", "raw")
        unzip = bool(cp.get("unzip", True))
        path.mkdir(parents=True, exist_ok=True)
        print(f"Competition (API): {cp_id} -> {path}")
        if not apply:
            continue
            
        retry_operation(
            api.competition_download_files, 
            cp_id, 
            path=str(path), 
            quiet=False, 
            force=True
        )
        if unzip:
            unzip_all_in_dir(path, path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Download datasets/competitions from Kaggle based on kaggle_datasets.json")
    
    # ALTERAÇÃO: O padrão agora busca o json na mesma pasta deste script (src/), e não na CWD
    default_cfg = Path(__file__).resolve().parent / "kaggle_datasets.json"
    
    ap.add_argument("--config", type=Path, default=default_cfg, help=f"Path to kaggle_datasets.json (default: {default_cfg})")
    ap.add_argument("--apply", action="store_true", help="Actually download files (default is dry-run)")
    ap.add_argument("--mode", choices=["cli", "api"], default="cli", help="Downloader mode")
    args = ap.parse_args(argv)

    if not args.config.exists():
        print(f"Config file not found: {args.config}")
        return 2

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg_dir = cfg.get("config", {}).get("kaggleConfigDir")
    cfg_dir_path = None
    if cfg_dir:
        cfg_dir_path = (ROOT / cfg_dir) if not Path(cfg_dir).is_absolute() else Path(cfg_dir)

    kjson = find_kaggle_json(cfg_dir_path)
    if not kjson:
        print("kaggle.json not found. Create it in %USERPROFILE%/.kaggle or in ./.kaggle and retry.")
        return 2
    set_kaggle_env(kjson)

    dry = not args.apply
    print(f"Dry-run: {dry}")
    print(f"Using KAGGLE_CONFIG_DIR={kjson.parent}")

    if args.mode == "api":
        try:
            download_datasets_api(cfg, apply=args.apply)
            download_competitions_api(cfg, apply=args.apply)
        except Exception as e:
            print(f"API mode failed: {e}")
            return 2
    else:
        runner = ensure_kaggle_cli()
        if not runner:
            print("Kaggle CLI not found. Install with: pip install kaggle")
            return 2
        download_datasets_cli(cfg, apply=args.apply, runner=runner)
        download_competitions_cli(cfg, apply=args.apply, runner=runner)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
