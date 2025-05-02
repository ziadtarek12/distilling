# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from evaluate import load

# Helper function to compute text accuracy
def compute_text_acc(preds, labels):
    correct = sum(1 for pred, label in zip(preds, labels) if pred.strip() == label.strip())
    return correct / len(preds)

# Helper function to evaluate equations and compute accuracy
def compute_equation_acc(preds, labels):
    correct = 0
    total = len(preds)
    for pred, label in zip(preds, labels):
        try:
            # Attempt to evaluate both prediction and label as Python expressions
            pred_val = eval(pred.strip())
            label_val = eval(label.strip())
            # Check for numerical equivalence (allowing for floating point inaccuracies)
            if isinstance(pred_val, (int, float)) and isinstance(label_val, (int, float)):
                if abs(pred_val - label_val) < 1e-6:
                    correct += 1
            # Check for string equivalence if not numerical
            elif str(pred_val) == str(label_val):
                 correct += 1
        except Exception:
            # If evaluation fails or types mismatch, compare as strings
            if pred.strip() == label.strip():
                correct += 1
    return correct / total

def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred # predictions shape (batch_size, seq_len), labels shape (batch_size, seq_len)

        # Replace -100 (ignore index) with pad_token_id
        # Ensure predictions are integers
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id).astype(np.int32)
        # Decode the whole batch
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels as well
        # Ensure labels are integers
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)
        # Decode the whole batch of labels
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute accuracy comparing the decoded strings using the helper function
        acc = compute_text_acc(decoded_preds, decoded_labels)

        return {'accuracy': acc}

    return compute_metrics

def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Replace -100 and ensure integer type for predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id).astype(np.int32)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) # Decode batch

        # Replace -100 and ensure integer type for labels
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) # Decode batch

        # Evaluate equations and compute accuracy using the helper function
        acc = compute_equation_acc(decoded_preds, decoded_labels)

        return {'accuracy': acc}

    return compute_metrics