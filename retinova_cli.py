#!/usr/bin/env python3
"""
RetiNova CLI — terminal-only wrapper that:
 - auto-uses default model (models/retinova_model.h5)
 - prompts user for an image path (drag & drop works)
 - runs full question bank per predicted condition (validated input)
 - produces Grad-CAM++ overlay and heatmap saved beside the input image
 - writes clickable file:// JSON report (retinova_results.json)
"""

import os
import sys
import json
import warnings
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


DEFAULT_MODEL_PATH = "models/retinova_model_tf2.h5"
CONF_THRESHOLD = 0.75  

class RetiNovaCLI:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, img_size=(224, 224), conf_threshold=CONF_THRESHOLD):
        self.img_size = img_size
        self.conf_threshold = float(conf_threshold)

        self.conditions = [
            "AMD",
            "Diabetic Retinopathy",
            "Glaucoma",
            "Hypertensive Retinopathy",
            "Normal",
            "Optical Retinopathy",
            "RVO",
            "Retinal Tears/Detachments",
        ]

        self.choice_map = {0: "Low", 1: "Mild", 2: "Moderate", 3: "High"}
        self.choice_conf_map = {"Low": 0.0, "Mild": 0.05, "Moderate": 0.1, "High": 0.2}

        try:
            self.model = load_model(model_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

        self.last_conv_layer_name = None
        try:
            for layer in reversed(self.model.layers):
                lname = layer.__class__.__name__.lower()
                if "conv" in lname or "separableconv" in lname or "depthwiseconv" in lname:
                    self.last_conv_layer_name = layer.name
                    break
            if self.last_conv_layer_name is None:
                for layer in reversed(self.model.layers):
                    if 'conv' in layer.name.lower():
                        self.last_conv_layer_name = layer.name
                        break
        except Exception:
            self.last_conv_layer_name = None

        self.risk_questions = {
            "AMD":[
                ("Are you over 55 years old?", ["Under 55","55-64","65-74","75 or older"]),
                ("Do you have difficulty reading or recognizing faces?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Have you noticed straight lines appear wavy or distorted?", ["No","Rarely","Occasionally","Frequently/severe"]),
                ("Is your central vision blurry or do you see a blind spot?", ["No","Slight blurring","Moderate blurring","Severe/sudden blind spot"]),
                ("Is it harder to see in low light?", ["No difficulty","Slight difficulty","Noticeable difficulty","Severe difficulty"]),
                ("Do you have a family history of macular degeneration?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Do you smoke or have you smoked?", ["Never","Rarely/occasional","Moderate","Heavy/long-term"]),
                ("Any recent sudden worsening of central vision?", ["No","Slight change","Noticeable gradual change","Sudden severe"])
            ],
            "Diabetic Retinopathy":[
                ("How long have you had diabetes?", ["<5 years","5-10 years","10-20 years",">20 years"]),
                ("How well is your blood sugar controlled?", ["Excellent","Good","Fair","Poor"]),
                ("Any recent changes in vision?", ["No","Slight","Moderate","Severe"]),
                ("Do you have high blood pressure?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Any history of kidney disease or vascular problems?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Have you noticed difficulties seeing at night?", ["No","Slight","Moderate","Severe"]),
                ("Have you been treated for eye problems before?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Are you taking medications for diabetes or eye issues?", ["No","Occasionally","Often","Very frequent/severe"])
            ],
            "Glaucoma":[
                ("Do you have a family history of glaucoma?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Have you noticed gradual loss of peripheral vision?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Have you experienced eye pain, halos, or headaches?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Has your vision worsened or become patchy?", ["No","Slightly","Moderate","Severe"]),
                ("Are you over 40 or have other risk factors?", ["No","Yes"]),
                ("Have you had trauma or previous eye surgery?", ["No","Yes"]),
                ("Any recent changes in prescription glasses?", ["No","Yes"])
            ],
            "Hypertensive Retinopathy":[
                ("Do you have high blood pressure?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Duration of high blood pressure?", ["<5 years","5-10 years","10-20 years",">20 years"]),
                ("Symptoms like blurry vision/headaches?", ["No","Slight","Moderate","Severe"]),
                ("Any smoking, diabetes, cholesterol, heart/kidney disease?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Recent blood pressure spikes?", ["No","Occasionally","Often","Very frequent/severe"]),
                ("Eye changes during routine checkups?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Experience double vision or spots?", ["No","Occasionally","Often","Very frequent/severe"])
            ],
            "RVO":[
                ("Sudden painless vision loss or blurring?", ["No","Slight","Moderate","Severe"]),
                ("New floaters, dark spots, shadows?", ["No","Slight","Moderate","Severe"]),
                ("Diagnosed with hypertension, diabetes, heart disease?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("History of stroke/clotting/high cholesterol?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Recent dizziness, weakness, neurological symptoms?", ["No","Slight","Moderate","Severe"]),
                ("Currently on blood-thinning/cardiovascular meds?", ["No","Occasionally","Often","Very frequent/severe"])
            ],
            "Retinal Tears/Detachments":[
                ("Sudden increase in floaters/flashes?", ["No","Slight","Moderate","Severe"]),
                ("Shadow or curtain affecting vision?", ["No","Slight","Moderate","Severe"]),
                ("Recent eye trauma or surgery?", ["No","Yes, minor","Yes, moderate","Yes, severe"]),
                ("Highly nearsighted?", ["No","Mild","Moderate","Severe"]),
                ("Distortion or blurring of vision?", ["No","Slight","Moderate","Severe"]),
                ("Family history of retinal detachment?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Diabetes or hypertension?", ["No","Yes, minor","Yes, moderate","Yes, severe"])
            ],
            "Optical Retinopathy":[
                ("Progressive vision loss or color issues?", ["No","Slight","Moderate","Severe"]),
                ("History of eye pain/neurological/systemic issues?", ["No","Slight","Moderate","Severe"]),
                ("Family history of vision loss or neurological disorders?", ["No","Distant relative","Parent/sibling","Multiple close family members"]),
                ("Recent infections, toxins, deficiencies?", ["No","Slight","Moderate","Severe"]),
                ("History of prior eye/brain surgery, trauma, or tumors?", ["No","Yes, minor","Yes, moderate","Yes, severe"])
            ],
            "Normal":[
                ("Any eye issues currently?", ["No","Slight","Moderate","Severe"])
            ]
        }

    def preprocess_image_from_path(self, img_path):
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise ValueError(f"Could not read image at {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(norm, axis=0)
        return batch, rgb.astype(np.uint8)

    def preprocess_image_from_bytes(self, img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes. Provide a valid image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(norm, axis=0)
        return batch, rgb.astype(np.uint8)

    def predict_condition(self, img_array):
        img_array = np.array(img_array, dtype=np.float32)
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)

        preds = self.model.predict(img_array, verbose=0)

        if isinstance(preds, (list, tuple)):
            preds = preds[-1]
        preds = np.array(preds)

        if preds.ndim == 1:
            preds = np.expand_dims(preds, axis=0)

        if preds.size == 1:
            idx = 0
            conf = float(preds.flatten()[0])
        else:
            idx = int(np.argmax(preds[0]))
            conf = float(preds[0, idx])

        condition = self.conditions[idx] if idx < len(self.conditions) else f"Class_{idx}"
        return condition, conf

    def make_gradcam_plus_plus(self, img_array, img_rgb, alpha=0.4):
        """
        Returns: (overlay_rgb_uint8, heatmap_bgr_uint8 or None)
        """
        try:
            img_np = np.array(img_array, dtype=np.float32)
            if img_np.ndim == 3:
                img_np = np.expand_dims(img_np, axis=0)
        except Exception:
            print("Grad-CAM++: could not coerce input; skipping heatmap.")
            return img_rgb, None

        if self.last_conv_layer_name is None:
            return img_rgb, None

        try:
            conv_layer = self.model.get_layer(self.last_conv_layer_name)
            grad_model = tf.keras.models.Model(inputs=self.model.input,
                                              outputs=[conv_layer.output, self.model.output])

            img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape() as tape1:
                    conv_outputs, predictions = grad_model(img_tensor, training=False)

                    if isinstance(predictions, (list, tuple)):
                        predictions = predictions[-1]

                    if tf.rank(predictions) == 0:
                        predictions = tf.reshape(predictions, (1, 1))
                    elif tf.rank(predictions) == 1:
                        predictions = tf.expand_dims(predictions, axis=0)

                    pred_index = tf.math.argmax(predictions[0])
                    class_channel = predictions[:, pred_index]

                grads = tape1.gradient(class_channel, conv_outputs)
            second_derivative = tape2.gradient(grads, conv_outputs)
            del tape2

            grads = grads[0]
            second_derivative = second_derivative[0]
            conv_outputs = conv_outputs[0]

            numerator = second_derivative
            denominator = 2.0 * second_derivative + tf.square(grads) + 1e-8
            alphas = tf.math.divide_no_nan(numerator, denominator)
            alphas = tf.nn.relu(alphas)

            weights = tf.reduce_sum(tf.maximum(grads, 0.0) * alphas, axis=(0, 1))

            heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
            heatmap = tf.maximum(heatmap, 0.0)

            max_val = tf.reduce_max(heatmap)
            max_val = tf.cast(max_val, tf.float32)
            if max_val > 0:
                heatmap = heatmap / (max_val + 1e-8)

            heatmap_uint8 = np.uint8(255 * heatmap.numpy())
            heatmap_resized = cv2.resize(heatmap_uint8, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
            heatmap_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
            overlay_rgb = cv2.addWeighted(img_rgb.astype(np.uint8), 0.6, heatmap_rgb, alpha, 0)

            return overlay_rgb.astype(np.uint8), heatmap_bgr.astype(np.uint8)

        except Exception as e:
            print("Grad-CAM++ failed:", e)
            return img_rgb, None
    def ask_risk_questions(self, condition, base_conf):
        questions = self.risk_questions.get(condition, [])
        answers = {}
        if not questions:
            return base_conf, answers

        print("\nAnswer the following questions to adjust confidence:")
        for i,(q,opts) in enumerate(questions):
            print(f"\n{i+1}. {q}")
            for j,opt in enumerate(opts):
                print(f"  {j}. {opt}")
            while True:
                try:
                    choice_raw = input("Enter choice number: ").strip()
                    choice = int(choice_raw)
                    if 0 <= choice < len(opts):
                        answers[q] = self.choice_map.get(choice,"Low")
                        break
                    else:
                        print("Invalid number. Try again.")
                except ValueError:
                    print("Invalid input. Enter a number.")

        final_conf = float(base_conf)
        for ans in answers.values():
            final_conf = min(1.0, final_conf + float(self.choice_conf_map.get(ans, 0)))
        return final_conf, answers

    def run_with_path(self, img_path):
        batch, rgb = self.preprocess_image_from_path(img_path)
        return self._run_core(batch, rgb, img_path)

    def run_with_bytes(self, img_bytes, input_label="stdin_image"):
        batch, rgb = self.preprocess_image_from_bytes(img_bytes)
        return self._run_core(batch, rgb, input_label)

    def _run_core(self, batch, rgb, input_path_or_label):
        condition, base_conf = self.predict_condition(batch)
        print(f"\nPredicted Condition: {condition}")
        print(f"Base Confidence: {base_conf*100:.2f}%")

        final_conf, answers = self.ask_risk_questions(condition, base_conf)

        if final_conf < self.conf_threshold:
            label = "Normal Eye"
            risk_note = f"AI confidence < {int(self.conf_threshold*100)}%. The eye appears mostly normal. However, you might be at some risk of {condition}. Consider regular checkups."
        else:
            label = condition
            risk_note = None

        print(f"\nFinal Label: {label}")
        print(f"Final Confidence: {final_conf*100:.2f}%")
        if answers:
            print("\nAnswers given:")
            for k,v in answers.items():
                print(f"  {k}: {v}")
        if risk_note:
            print("\nRisk Note:", risk_note)

        overlay_rgb, heatmap_bgr = self.make_gradcam_plus_plus(batch, rgb)

        if os.path.exists(input_path_or_label):
            input_abs = os.path.abspath(input_path_or_label)
            in_dir = os.path.dirname(input_abs)
            base = os.path.splitext(os.path.basename(input_abs))[0]
        else:
            in_dir = os.getcwd()
            base = input_path_or_label.replace(os.sep, "_").replace(":", "")

        overlay_name = f"{base}_gradcam_overlay.png"
        heatmap_name = f"{base}_gradcam_heatmap.png"
        overlay_path = os.path.join(in_dir, overlay_name)
        heatmap_path = os.path.join(in_dir, heatmap_name)

        try:
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            print(f"\nGrad-CAM++ overlay saved as '{overlay_path}'")
        except Exception as e:
            print("Failed to save overlay:", e)

        if heatmap_bgr is not None:
            try:
                cv2.imwrite(heatmap_path, heatmap_bgr)
                print(f"Grad-CAM++ heatmap saved as '{heatmap_path}'")
            except Exception as e:
                print("Failed to save heatmap:", e)
        else:
            heatmap_path = None

        def to_file_link(p):
            if not p:
                return None
            p_abs = os.path.abspath(p)
            return "file:///" + p_abs.replace(os.sep, "/")

        results = {
            "model_path": os.path.abspath(self.model.optimizer._name) if hasattr(self.model, "optimizer") and self.model.optimizer is not None else os.path.abspath(DEFAULT_MODEL_PATH),
            "input_image": to_file_link(os.path.abspath(input_path_or_label)) if os.path.exists(input_path_or_label) else None,
            "predicted_condition": condition,
            "base_confidence": float(base_conf),
            "final_confidence": float(final_conf),
            "final_label": label,
            "risk_note": risk_note,
            "answers": answers,
            "gradcam_overlay_path": to_file_link(overlay_path),
            "gradcam_heatmap_path": to_file_link(heatmap_path) if heatmap_path else None
        }
        out_json = os.path.join(in_dir, "retinova_results.json")
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nJSON report saved to '{out_json}'")
        except Exception as e:
            print("Failed to write JSON report:", e)

        return results
def main():
    model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model not found at default '{model_path}'. Provide model path as first arg.")
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            model_path = sys.argv[1]
        else:
            print("Usage: python retinova_cli.py [model.h5] [image_path]")
            sys.exit(1)

    arg_offset = 0
    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]) and sys.argv[1].lower().endswith(".h5"):
        model_path = sys.argv[1]
        arg_offset = 1
    pipeline = RetiNovaCLI(model_path=model_path)
    image_arg = None
    if len(sys.argv) > arg_offset + 1:
        image_arg = sys.argv[arg_offset + 1]

    if not image_arg:
        image_arg = input("Enter path to image (drag & drop works): ").strip().strip('"').strip("'")
    if image_arg == "-":
        stdin_bytes = sys.stdin.buffer.read()
        if not stdin_bytes:
            print("No data read from stdin. Example to pipe on Unix: cat image.png | python retinova_cli.py -")
            sys.exit(1)
        pipeline.run_with_bytes(stdin_bytes, input_label="stdin_image")
        return
    if os.path.exists(image_arg) and os.path.isfile(image_arg):
        pipeline.run_with_path(image_arg)
        return
    try:
        b = base64.b64decode(image_arg)
        pipeline.run_with_bytes(b, input_label="pasted_base64_image")
        return
    except Exception:
        print("Input was not a file path and not valid base64. Please rerun and provide a valid image path (drag & drop works).")
        sys.exit(1)


if __name__ == "__main__":
    main()
