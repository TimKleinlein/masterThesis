import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import logging
import torch
import os
import pickle as pkl
from transformers import TrainerCallback, TrainerState, TrainerControl
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import matplotlib.pyplot as plt
import json

# Set the environment variable to use only GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the base directory for outputs
base_output_dir = "data/bert_classifier"
os.makedirs(base_output_dir, exist_ok=True)

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_output_dir, "trainlog.txt")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Redirect print statements to the logger
class PrintLogger:
    def write(self, message):
        if message.strip():  # Ignore empty messages
            logger.info(message.strip())

    def flush(self):
        pass  # No need to implement this for the logger

sys.stdout = PrintLogger()

# Load transcriptions data
with open('data/transcriptions.pkl', 'rb') as file:
    transcriptions = pkl.load(file)

accuracy_modes = ["low_acc", "mid_acc", "high_acc"]

for acc_mode in accuracy_modes:
    logger.info(f"Starting processing for accuracy mode: {acc_mode}")

    # Prepare data for the current accuracy mode
    data = []
    groups = []
    for dr in transcriptions.keys():
        for ind, utt in enumerate(transcriptions[dr][acc_mode]):
            data.append({'text': utt, 'label': transcriptions[dr]['role'], 'group': dr})

    ROLES_ToR = {
        "CREWMATE": ["crewmate", "mayor", "medium", "swapper", "time master", "engineer", "sheriff", "deputy",
                     "lighter",
                     "detective", "medic", "seer", "hacker", "tracker", "snitch", "spy", "portalmaker",
                     "security guard",
                     "medium", "trapper", "nice guesser", "bait", "shifter"],
        "NEUTRAL": ["neutral", "jester", "arsonist", "jackal", "sidekick", "vulture", "lawyer", "pursuer", "thief"],
        "IMPOSTOR": ["impostor", "godfather", "mafioso", "janitor", "morphling", "camouflager", "vampire", "eraser",
                     "trickster", "cleaner", "warlock", "bounty hunter", "witch", "ninja", "bomber", "yo-yo",
                     "evil guesser"]
    }

    ROLES_ToU = {
        "CREWMATE": ["crewmate", "detective", "haunter", "investigator", "mystic", "seer", "snitch", "spy", "tracker",
                     "trapper", "sheriff", "veteran", "vigilante", "altruist", "medic", "engineer", "mayor", "medium",
                     "swapper", "transporter", "aurial", "hunter", "imitator", "oracle", "vampire hunter", "time lord"],
        "NEUTRAL": ["neutral", "amnesiac", "guardian angel", "survivor", "executioner", "jester", "phantom", "arsonist",
                    "plaguebearer", "the glitch", "werewolf", "doomsayer", "juggernaut"],
        "IMPOSTOR": ["impostor", "grenadier", "morphling", "swooper", "traitor", "blackmailer", "janitor", "miner",
                     "undertaker", "bomber", "escapist", "venerer", "warlock", "poisoner", "underdog"]
    }

    for entry in data:
        found = False
        for r in ROLES_ToR.keys():
            for sr in ROLES_ToR[r]:
                if sr == entry['label']:
                    entry['label'] = r
                    found = True
                    break
        if not found:
            for r in ROLES_ToU.keys():
                for sr in ROLES_ToU[r]:
                    if sr == entry['label']:
                        entry['label'] = r
                        found = True
                        break

    # Remove neutral roles from data
    data = [entry for entry in data if entry['label'] != 'NEUTRAL']

    texts = [entry['text'] for entry in data]
    labels = [0 if entry['label'] == 'CREWMATE' else 1 for entry in data]
    groups = [entry['group'] for entry in data]

    # Create a Dataset from the dictionary
    dataset = datasets.Dataset.from_dict({'text': texts, 'label': labels, 'group': groups})

    # Tokenize the data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)


    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Load metric
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    class CustomCallback(TrainerCallback):
        def __init__(self, test_dataset):
            self.test_dataset = test_dataset

        def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            # Get the predictions and labels for the evaluation dataset
            predictions, labels, _ = trainer.predict(self.test_dataset)
            predictions = np.argmax(predictions, axis=1)
            crewmate_count = np.sum(predictions == 0)
            impostor_count = np.sum(predictions == 1)

            logger.info(
                f"Epoch {state.epoch}: CREWMATE predicted {crewmate_count} times, IMPOSTOR predicted {impostor_count} times")


    # Initialize group k-fold cross-validation
    gkf = GroupKFold(n_splits=5)
    fold = 0

    # Iterate through each fold
    for train_index, test_index in gkf.split(tokenized_dataset, groups=groups):
        fold += 1
        logger.info(f"Starting fold {fold} for accuracy mode: {acc_mode}")

        train_dataset = tokenized_dataset.select(train_index)
        test_dataset = tokenized_dataset.select(test_index)

        # Create directory for accuracy mode and fold if it doesn't exist
        fold_dir = os.path.join(base_output_dir, acc_mode)
        os.makedirs(fold_dir, exist_ok=True)

        # Save test indices for the current fold
        test_indices_path = os.path.join(fold_dir, f"test_indices_fold_{fold}.json")
        with open(test_indices_path, 'w') as f:
            json.dump(test_index.tolist(), f)

        # Convert to lists for oversampling
        train_texts = train_dataset['text']
        train_labels = train_dataset['label']

        # Apply oversampling only on the training data
        oversampler = RandomOverSampler(sampling_strategy='minority')
        train_texts_resampled, train_labels_resampled = oversampler.fit_resample(np.array(train_texts).reshape(-1, 1),
                                                                                 np.array(train_labels))
        train_texts_resampled = train_texts_resampled.flatten()

        # Create a new dataset from the resampled data
        train_data_resampled = {'text': train_texts_resampled.tolist(), 'label': train_labels_resampled.tolist()}
        train_dataset_resampled = datasets.Dataset.from_dict(train_data_resampled)
        tokenized_train_dataset_resampled = train_dataset_resampled.map(tokenize_function, batched=True)

        # Reload the model for each fold
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        model.to(device)  # Ensure the model is on the GPU

        # Define training arguments
        output_dir = os.path.join(base_output_dir, acc_mode, f"test_trainer_fold_{fold}")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=2e-5,  # Adjusted learning rate
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            report_to="none",  # Disable reporting to external services
            load_best_model_at_end=True,
            save_strategy="epoch",  # Save model at the end of each epoch
            save_total_limit=1,  # Limit the total number of checkpoints
        )

        # Initialize the Trainer with the CustomCallback that takes the current test_dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset_resampled,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            callbacks=[CustomCallback(test_dataset)]  # Add the custom callback
        )

        # Train the model
        trainer.train()

        # Save the best model and tokenizer for each fold
        model_fold_path = os.path.join(base_output_dir, acc_mode, f"best_model_fold_{fold}")
        trainer.save_model(model_fold_path)
        tokenizer.save_pretrained(model_fold_path)

        # Evaluate the model
        eval_results = trainer.evaluate()

        # Save the evaluation results
        eval_results_path = os.path.join(base_output_dir, acc_mode, f"evaluation_results_fold_{fold}.txt")
        os.makedirs(os.path.dirname(eval_results_path), exist_ok=True)
        with open(eval_results_path, "w") as writer:
            for key, value in eval_results.items():
                writer.write(f"{key}: {value}\n")

        logger.info(f"Evaluation results for fold {fold} saved to {eval_results_path}")

        # Generate and save confusion matrix
        predictions, labels, _ = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(labels, predictions, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CREWMATE', 'IMPOSTOR'])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.subplots_adjust(left=0.2)  # Adjust the padding to ensure the y-axis label is not cut off
        cm_path = os.path.join(base_output_dir, acc_mode, f"confusion_matrix_fold_{fold}.png")
        plt.savefig(cm_path)
        plt.close()

        logger.info(f"Confusion matrix for fold {fold} saved to {cm_path}")

    logger.info(f"Completed processing for accuracy mode: {acc_mode}")

logger.info("Cross-validation for all accuracy modes completed.")


# exclude later discussion rounds from analysis because players often already dead? (6829 - 6239 (<3) - 4967 (<2))
