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
    # Ensure inputs are lists of strings
    preds = [str(p) for p in preds]
    labels = [str(l) for l in labels]
    correct = sum(1 for pred, label in zip(preds, labels) if pred.strip() == label.strip())
    if not preds: # Avoid division by zero
        return 0.0
    return correct / len(preds)

# Helper function to evaluate equations and compute accuracy
def compute_equation_acc(preds, labels):
    correct = 0
    # Ensure inputs are lists of strings
    preds = [str(p) for p in preds]
    labels = [str(l) for l in labels]
    total = len(preds)
    if total == 0: # Avoid division by zero
        return 0.0
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
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id

        # Handle predictions being a list (e.g., in task_prefix model)
        if isinstance(predictions, list):
            # Extract the predictions for the 'pred' task
            predictions = predictions[0]
        
        # --- Predictions Preprocessing ---
        # Replace -100 (ignore index)
        predictions = np.where(predictions == -100, pad_token_id, predictions)
        # Check for NaN/Inf and replace with pad_token_id
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: NaN or Inf detected in predictions. Replacing with pad_token_id.")
            predictions = np.nan_to_num(predictions, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        # Clip predictions to valid token ID range
        predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int32)
        # Decode the whole batch
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Handle labels being a list (e.g., in task_prefix model)
        if isinstance(labels, list):
            # Extract the labels for the 'pred' task
            labels = labels[0]
            
        # --- Labels Preprocessing ---
        # Replace -100 in labels as well
        labels = np.where(labels == -100, pad_token_id, labels)
        # Check for NaN/Inf in labels (less likely but good practice)
        if np.isnan(labels).any() or np.isinf(labels).any():
            print("Warning: NaN or Inf detected in labels. Replacing with pad_token_id.")
            labels = np.nan_to_num(labels, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        # Clip labels to valid token ID range
        labels = np.clip(labels, 0, vocab_size - 1).astype(np.int32)
        # Decode the whole batch of labels
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute accuracy comparing the decoded strings using the helper function
        acc = compute_text_acc(decoded_preds, decoded_labels)

        return {'accuracy': acc}

    return compute_metrics

def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id

        # Handle predictions being a list (e.g., in task_prefix model)
        if isinstance(predictions, list):
            # Extract the predictions for the 'pred' task
            predictions = predictions[0]
            
        # --- Predictions Preprocessing ---
        # Replace -100 (ignore index)
        predictions = np.where(predictions == -100, pad_token_id, predictions)
        # Check for NaN/Inf and replace with pad_token_id
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: NaN or Inf detected in predictions. Replacing with pad_token_id.")
            predictions = np.nan_to_num(predictions, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        # Clip predictions to valid token ID range
        predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int32)
        # Decode the whole batch
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Handle labels being a list (e.g., in task_prefix model)
        if isinstance(labels, list):
            # Extract the labels for the 'pred' task
            labels = labels[0]
            
        # --- Labels Preprocessing ---
        # Replace -100 in labels as well
        labels = np.where(labels == -100, pad_token_id, labels)
        # Check for NaN/Inf in labels
        if np.isnan(labels).any() or np.isinf(labels).any():
            print("Warning: NaN or Inf detected in labels. Replacing with pad_token_id.")
            labels = np.nan_to_num(labels, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        # Clip labels to valid token ID range
        labels = np.clip(labels, 0, vocab_size - 1).astype(np.int32)
        # Decode the whole batch of labels
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Evaluate equations and compute accuracy using the helper function
        acc = compute_equation_acc(decoded_preds, decoded_labels)

        return {'accuracy': acc}

    return compute_metrics

# Function for task_prefix model with text data - evaluate both label and rationale tasks
def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        # For standard model, prediction is just a single tensor for label prediction
        predictions, labels = eval_pred
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id
        
        # Ensure predictions is 2D (batch_size, seq_len)
        if isinstance(predictions, list):
            # For some model outputs, predictions might be a list of tensors
            predictions = predictions[0]  # Take the first one (label predictions)
        
        # Preprocess predictions (same preprocessing as in compute_metrics_text)
        predictions = np.where(predictions == -100, pad_token_id, predictions)
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: NaN or Inf detected in predictions. Replacing with pad_token_id.")
            predictions = np.nan_to_num(predictions, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int32)
        
        # Ensure predictions is a proper 2D array for batch_decode
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)
            
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Handle labels similarly
        if isinstance(labels, list):
            labels = labels[0]  # Take the first one (main task labels)
            
        # Preprocess labels
        labels = np.where(labels == -100, pad_token_id, labels)
        if np.isnan(labels).any() or np.isinf(labels).any():
            print("Warning: NaN or Inf detected in labels. Replacing with pad_token_id.")
            labels = np.nan_to_num(labels, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        labels = np.clip(labels, 0, vocab_size - 1).astype(np.int32)
        
        # Ensure labels is a proper 2D array for batch_decode
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)
            
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate accuracy
        acc = compute_text_acc(decoded_preds, decoded_labels)
        
        return {'accuracy': acc}
    
    return compute_metrics

# Function for task_prefix model with equation data - evaluate both label and rationale tasks
def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        # For standard model, prediction is just a single tensor for label prediction
        predictions, labels = eval_pred
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id
        
        # Ensure predictions is 2D (batch_size, seq_len)
        if isinstance(predictions, list):
            # For some model outputs, predictions might be a list of tensors
            predictions = predictions[0]  # Take the first one (label predictions)
        
        # Preprocess predictions (same preprocessing as in compute_metrics_equation)
        predictions = np.where(predictions == -100, pad_token_id, predictions)
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("Warning: NaN or Inf detected in predictions. Replacing with pad_token_id.")
            predictions = np.nan_to_num(predictions, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        predictions = np.clip(predictions, 0, vocab_size - 1).astype(np.int32)
        
        # Ensure predictions is a proper 2D array for batch_decode
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(1, -1)
            
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Handle labels similarly
        if isinstance(labels, list):
            labels = labels[0]  # Take the first one (main task labels)
            
        # Preprocess labels
        labels = np.where(labels == -100, pad_token_id, labels)
        if np.isnan(labels).any() or np.isinf(labels).any():
            print("Warning: NaN or Inf detected in labels. Replacing with pad_token_id.")
            labels = np.nan_to_num(labels, nan=pad_token_id, posinf=pad_token_id, neginf=pad_token_id)
        labels = np.clip(labels, 0, vocab_size - 1).astype(np.int32)
        
        # Ensure labels is a proper 2D array for batch_decode
        if len(labels.shape) == 1:
            labels = labels.reshape(1, -1)
            
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate equation accuracy
        acc = compute_equation_acc(decoded_preds, decoded_labels)
        
        return {'accuracy': acc}
    
    return compute_metrics