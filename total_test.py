import os
import subprocess
import torch
import sys

# ============================================================
# 1. Hyperparameters & Settings (논문 Section 4.1 기반)
# ============================================================
COMMON = {
    "encoder_rnn_cell": "LSTM", "decoder_rnn_cell": "LSTM",
    "encoder_num_layers": 4, "decoder_num_layers": 4,
    "encoder_hidden_size": 1000, "decoder_hidden_size": 1000,
    "embedding_size": 1000, "batch_size": 128,
    "learning_rate": 1.0, "gradient_clip": 5.0,
}

EPOCHS_BASE, EPOCHS_DR = 10, 12
DECAY_BASE, DECAY_DR = 5, 8

# ============================================================
# 2. Full Experiment Definitions
# ============================================================
experiments = {

    # ========================================================
    # TABLE 1 — WMT14 En→De (Main Results)
    # ========================================================
    
    # [Row 1] Base Model
    # 설정: Reverse=False, Dropout=0.0
    "T1_Base": {
        "dataset": "wmt14-en-de", 
        "args": {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE, 
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0, 
                 "attention_type": "none", 
                 "reverse": False}
    },

    # [Row 2] Base + Reverse
    # 설정: Reverse=True
    "T1_Base_Reverse": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type": "none",
                 "reverse": True}
    },

    # [Row 3] Base + Reverse + Dropout
    # 설정: Dropout=0.2 (Epochs=12, Decay=8)
    "T1_Base_Reverse_Dropout": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "none",
                 "reverse": True}
    },

    # [Row 4] + Global (Location)
    # 논문 Table 1 Row 4: Global Attention (location)
    "T1_Global_Location": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": False, # Attentional models typically use input feeding
                 "reverse": True}
    },

    # [Row 5] + Global (Location) + Feed
    # (Row 4와 동일하게 Feed=True로 설정됨. 논문에서는 Feed 유무로 성능 차이를 강조함)
    "T1_Global_Location_Feed": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    # [Row 6] + Local-p (General) + Feed
    "T1_LocalP_General_Feed": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-p", "attention_score": "general",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    # [Row 7] + Local-p (General) + Feed + Unk Replace
    # 학습 설정은 Row 6와 동일하며, 평가 시 UNK Replacement 수행 (Post-process)
    "T1_LocalP_General_Feed_Unk": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-p", "attention_score": "general",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed": True,
                 "reverse": True},
        "post_process": "unk_replace"
    },
    
    # [Row 8] Ensemble (Inference Only)
    "T1_Ensemble8_Unk": {
        "dataset": "wmt14-en-de",
        "ensemble": True,
        "members": ["T1_Global_Location_Feed", "T1_LocalP_General_Feed"], # 실제로는 8개 모델 필요
        "post_process": "unk_replace"
    },


    # ========================================================
    # TABLE 3 — WMT15 De→En (Direction Change)
    # ========================================================
    
    "T3_Base_Reverse": {
        "dataset": "wmt15-deen", # Triggers De->En in dataset.py
        "args": {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type": "none",
                 "reverse": True}
    },

    "T3_Global_Location": {
        "dataset": "wmt15-deen",
        "args": {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T3_Global_Location_Feed": {
        "dataset": "wmt15-deen",
        "args": {**COMMON, "max_epochs": EPOCHS_BASE, "lr_decay_start": DECAY_BASE,
                 "encoder_rnn_dropout": 0.0, "decoder_rnn_dropout": 0.0,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T3_Global_Dot_Drop_Feed": {
        "dataset": "wmt15-deen",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "dot",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T3_Global_Dot_Drop_Feed_Unk": {
        "dataset": "wmt15-deen",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "dot",
                 "luong_input_feed": True,
                 "reverse": True},
        "post_process": "unk_replace"
    },


    # ========================================================
    # TABLE 4 — Attention Ablation Study (En→De)
    # Common: Reverse=True, Dropout=0.2, Feed=True, Epochs=12
    # ========================================================

    "T4_Global_Location": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "location",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_Global_Dot": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "dot",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_Global_General": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "global", "attention_score": "general",
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalM_Dot": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-m", "attention_score": "dot",
                 "half_window_size": 10,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalM_General": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-m", "attention_score": "general",
                 "half_window_size": 10,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalP_Dot": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-p", "attention_score": "dot",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed": True,
                 "reverse": True}
    },

    "T4_LocalP_General": {
        "dataset": "wmt14-en-de",
        "args": {**COMMON, "max_epochs": EPOCHS_DR, "lr_decay_start": DECAY_DR,
                 "encoder_rnn_dropout": 0.2, "decoder_rnn_dropout": 0.2,
                 "attention_type": "local-p", "attention_score": "general",
                 "half_window_size": 10, "local_p_hidden_size": 1000,
                 "luong_input_feed": True,
                 "reverse": True}
    }
}


# ============================================================
# 3. Execution Engine
# ============================================================
def run_command(cmd, log_path):
    print(f"[Exec] {cmd}")
    # 로그 파일 디렉토리 생성
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "w") as f:
        # 프로세스 실행, 표준 출력/에러를 로그 파일로 리다이렉션
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, check=True)
def find_latest_checkpoint(save_path):
    """
    Find the latest checkpoint file in save_path.
    
    Returns:
        (checkpoint_path, epoch) or (None, 0) if no checkpoint found
    """
    if not os.path.exists(save_path):
        return None, 0
    
    checkpoint_files = [f for f in os.listdir(save_path) if f.startswith('model_epoch') and f.endswith('.pt')]
    
    if not checkpoint_files: 
        return None, 0
    
    # Extract epoch numbers and find max
    epochs = []
    for f in checkpoint_files:
        try:
            # model_epoch10.pt -> 10
            epoch_num = int(f.replace('model_epoch', '').replace('.pt', ''))
            epochs.append((epoch_num, f))
        except ValueError:
            continue
    
    if not epochs:
        return None, 0
    
    latest_epoch, latest_file = max(epochs, key=lambda x: x[0])
    return os.path. join(save_path, latest_file), latest_epoch

def main():
    print("======================================================")
    print("      RNN-NMT Reproduction: Total Test Runner         ")
    print("======================================================")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        common_flags = " --cuda"
    else:
        print("CUDA NOT Available. Running on CPU (Warning: Slow)")
        common_flags = ""

    os.makedirs("checkpoints", exist_ok=True)

    for exp_name, config in experiments.items():
        print(f"\n>>> Processing Experiment: {exp_name}")
        
        # 1. Skip Ensemble Training (Requires separately trained models or custom logic)
        if config.get("ensemble", False):
            print(f"Skipping training for Ensemble '{exp_name}'. (Inference logic required)")
            continue

        # 2. Setup Directories
        save_path = os.path.join("checkpoints", exp_name)
        os.makedirs(save_path, exist_ok=True)
        log_file = os.path.join(save_path, "train.log")
        
        # 3. Construct Training Command
        # 기본 플래그 구성
        cmd = f"python train.py --dataset {config['dataset']} --save-path {save_path}" + common_flags
        
        # 실험별 인자 추가
        for key, value in config['args'].items():
            # Python dict key -> CLI flag 변환 (예: lr_decay_start -> --lr-decay-start)
            flag_name = "--" + key.replace("_", "-")
            
            if isinstance(value, bool):
                if value: cmd += f" {flag_name}"
            else:
                cmd += f" {flag_name} {value}"

        # 4. Execute Training
        try:
            # 이미 학습 완료된 경우(args 파일 존재 시) 스킵
            if not os.path.exists(os.path.join(save_path, "args")): 
                print(f"Training started... Logs: {log_file}")
                run_command(cmd, log_file)
                print("Training completed successfully.")
            else:
                print("Checkpoint found. Skipping training phase.")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error during training {exp_name}. See logs at {log_file}")
            continue
        # ==================== 수정: Resume 로직 추가 ====================

        # 1. 최신 체크포인트 확인
        checkpoint_path, completed_epochs = find_latest_checkpoint(save_path)
        max_epochs = config['args']['max_epochs']
        
        # 2. 이미 학습 완료된 경우 스킵
        if completed_epochs >= max_epochs:
            print(f"Training already completed ({completed_epochs}/{max_epochs} epochs). Skipping.")
        else:
            # 3. 학습 명령 구성
            cmd = f"python train.py --dataset {config['dataset']} --save-path {save_path}" + common_flags
            
            # Resume 플래그 추가
            if checkpoint_path: 
                cmd += f" --resume {checkpoint_path}"
                print(f"Resuming from epoch {completed_epochs}/{max_epochs}")
            else:
                print(f"Starting fresh training (0/{max_epochs} epochs)")
            
            # 실험별 인자 추가
            for key, value in config['args'].items():
                flag_name = "--" + key.replace("_", "-")
                
                if isinstance(value, bool):
                    if value:  cmd += f" {flag_name}"
                else:
                    cmd += f" {flag_name} {value}"

            # 4. 학습 실행
            try:
                print(f"Training started...  Logs:  {log_file}")
                run_command(cmd, log_file)
                print("Training completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"! !! Error during training {exp_name}. See logs at {log_file}")
                continue
        # ================================================================

        # 5. Evaluation (Calculate BLEU)
        # 평가를 위한 Reference File 결정 (dataset.py의 로직과 일치해야 함)
        # dataset.py는 무조건 'base' 폴더를 보고, 방향만 바꿈.
        # En->De : Target is German (.de)
        # De->En : Target is English (.en)
        is_deen = 'deen' in config['dataset']
        tgt_ext = 'en' if is_deen else 'de'
        
        # Reference file path: data/wmt14_vocab50k/base/test.{de|en} or data/wmt15_vocab50k/base/test.{de|en}
        is_wmt15 = 'wmt15' in config['dataset'].lower()
        dataset_name = 'wmt15' if is_wmt15 else 'wmt14'
        ref_file = f"data/{dataset_name}_vocab50k/base/test.{tgt_ext}"
        
        # 평가 로그 파일
        eval_log_file = os.path.join(save_path, "eval.log")
        
        # BLEU 계산 명령어
        # calculate_bleu.py가 --model-path 디렉토리 내의 epoch에 맞는 모델을 로드한다고 가정
        # max_epochs 값을 epoch 인자로 전달
        eval_cmd = f"python calculate_bleu.py --model-path {save_path} --reference-path {ref_file} --epoch {config['args']['max_epochs']}" + common_flags
        
        print(f"Evaluating BLEU... Logs: {eval_log_file}")
        try:
            run_command(eval_cmd, eval_log_file)
            
            # 결과 출력 (로그 파일의 마지막 줄에 BLEU 점수가 있다고 가정)
            with open(eval_log_file, 'r') as f:
                lines = f.readlines()
                if lines: 
                    print(f"Result [{exp_name}]: {lines[-1].strip()}")
                else:
                    print("Warning: Eval log is empty.")
                    
        except subprocess.CalledProcessError:
            print(f"Evaluation failed for {exp_name}. Check logs.")
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()
