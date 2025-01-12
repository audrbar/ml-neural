import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments, \
    AutoModelForSequenceClassification
from sklearn.datasets import fetch_20newsgroups


# Tokenization Function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


# Custom Model
class CustomDistilBertModel(nn.Module):
    def __init__(self):
        super(CustomDistilBertModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


# Load Dataset
newsgroups_data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'])
X, y = newsgroups_data.data, newsgroups_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenized_datasets = dataset.map(tokenize_function, batched=True)


# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Use Hugging Face Pre-trained Model with Custom Classifier
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)
# model = CustomDistilBertModel()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# Train Model
trainer.train()

# Evaluate Model
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']:.2f}")
