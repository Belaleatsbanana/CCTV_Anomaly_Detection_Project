import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Adjust these imports to match your package structure
from resnext import generate_model       # ResNeXt I3D backbone
from models import Learner              # your Learner architecture

def load_feature_extractor(weight_path, device):
    """
    Loads the ResNeXt-I3D backbone and its pretrained Kinetics weights.
    """
    model = generate_model()  # returns DataParallel(model).cuda()
    checkpoint = torch.load(
        weight_path,
        map_location=device,
        weights_only=False
    )
    sd = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


def load_classifier(weight_path, input_dim, device):
    """
    Loads your trained Learner classifier.
    """
    clf = Learner(input_dim=input_dim).to(device)
    checkpoint = torch.load(
        weight_path,
        map_location=device,
        weights_only=False
    )
    clf.load_state_dict(checkpoint)
    clf.eval()
    return clf


class VideoTransform:
    """
    Preprocesses video frames: resize, to tensor, normalize.
    """
    def __init__(self, size=112):
        self.pipeline = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, frame):
        return self.pipeline(frame)


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (w, h)


def sliding_windows(frames, clip_len=16):
    for start in range(0, len(frames) - clip_len + 1):
        clip = frames[start:start + clip_len]
        yield start, clip


def infer_scores(frames, feat_model, clf, transform, device):
    n = len(frames)
    scores = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)

    with torch.no_grad():
        for start, clip in sliding_windows(frames, clip_len=16):
            tensor = torch.stack([transform(f) for f in clip], dim=1).unsqueeze(0).to(device)
            logits, feat = feat_model(tensor)
            if start == 0:
                print(f"DEBUG: feat shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}")
                np.save("debug_feat_window0.npy", feat.cpu().numpy())
                print("DEBUG: saved debug_feat_window0.npy")
            feat = F.normalize(feat, p=2, dim=1)
            score = clf(feat).item()
            scores[start:start + 16] += score
            counts[start:start + 16] += 1

    valid = counts > 0
    scores[valid] /= counts[valid]
    return scores


def annotate_and_write(frames, scores, fps, size, out_path, threshold):
    # Ensure output directory exists
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    # Initialize writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {out_path}")

    print(f"Writing {len(frames)} frames to {out_path}")
    for i, frame in enumerate(frames):
        sc = scores[i]
        txt = f"Pred:{sc:.2f}"
        color = (0, 0, 255) if sc > threshold else (200, 200, 200)
        cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if sc > threshold:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        writer.write(frame)
    writer.release()


def plot_scores(scores, fps, out_path):
    import matplotlib.pyplot as plt
    times = np.arange(len(scores)) / fps
    plt.figure()
    plt.plot(times, scores)
    plt.xlabel('Time (s)')
    plt.ylabel('Anomaly Score')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Anomaly detection on video using pre-trained I3D + custom Learner")
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path or directory for output')
    parser.add_argument('--feat', default='RGB_Kinetics_16f.pth', help='Path to feature extractor weights')
    parser.add_argument('--clf', default='best_model_auc0.8464.pth', help='Path to classifier weights')
    parser.add_argument('--threshold', type=float, default=0.4, help='Anomaly score threshold')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare output paths
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    # If output is directory, create annotated file inside
    if os.path.isdir(args.output) or args.output.endswith(os.sep):
        os.makedirs(args.output, exist_ok=True)
        out_video = os.path.join(args.output, input_basename + '_annotated.mp4')
    else:
        out_video = args.output

    # Load models
    feat_model = load_feature_extractor(args.feat, device)
    clf_model = load_classifier(args.clf, input_dim=2048, device=device)

    # Read video
    frames, fps, size = read_frames(args.input)

    # Inference
    transform = VideoTransform(size=112)
    scores = infer_scores(frames, feat_model, clf_model, transform, device)

    # Write outputs
    annotate_and_write(frames, scores, fps, size, out_video, args.threshold)
    scores_path = os.path.splitext(out_video)[0] + '_scores.png'
    plot_scores(scores, fps, scores_path)

    print(f"Annotated video saved to: {out_video}")
    print(f"Score curve saved to:    {scores_path}")

if __name__ == '__main__':
    main()
