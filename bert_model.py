import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from torch.optim import AdamW
from torch.nn import Dropout, CrossEntropyLoss
import os

# GitHub'dan hata düzeltme commit'lerini çeken fonksiyon
def fetch_commits(repo_name, auth_token, max_pages=10):
    commits = []
    page = 1

    while page <= max_pages:
        commits_url = f"https://api.github.com/repos/{repo_name}/commits?page={page}&per_page=100"
        headers = {'Authorization': f'token {auth_token}'}
        response = requests.get(commits_url, headers=headers)
        if response.status_code == 200:
            page_commits = response.json()
            commits.extend([commit for commit in page_commits if 'fix' in commit['commit']['message'].lower()])
            if len(page_commits) < 100:
                break
        else:
            print("Failed to fetch commits, status code:", response.status_code)
            break
        page += 1

    print(f"Fetched {len(commits)} commits that mention 'fix'")
    return commits

# Commit'lerden kod değişikliklerini çıkaran fonksiyon
def extract_code_changes(commits, auth_token):
    headers = {'Authorization': f'token {auth_token}'}
    code_changes = []

    for commit in commits:
        diff_url = commit['html_url'] + '.diff'
        response = requests.get(diff_url, headers=headers)
        if response.status_code == 200:
            diff_data = response.text
            changes = parse_diff(diff_data)
            code_changes.extend(changes)
        else:
            print(f"Failed to fetch diff for commit {commit['sha']}, status code:", response.status_code)

    return code_changes

# Diff verilerinden kod bloklarını ayıklayan fonksiyon
def parse_diff(diff_data):
    changes = []
    lines = diff_data.split('\n')
    old_code = []
    new_code = []

    for line in lines:
        if line.startswith('-') and not line.startswith('---'):
            old_code.append(line[1:].strip())
        elif line.startswith('+') and not line.startswith('+++'):
            new_code.append(line[1:].strip())

    if old_code and new_code:
        changes.append((' '.join(old_code), ' '.join(new_code)))

    return changes

# Veri setini hazırlama ve model eğitimi için fonksiyon
def prepare_dataset(data, tokenizer):
    texts = [change[0] for change in data]
    labels = torch.tensor([1] * len(data))


    neg_texts = ["no changes"] * len(data)
    neg_labels = torch.tensor([0] * len(data))

    texts.extend(neg_texts)
    labels = torch.cat((labels, neg_labels), dim=0)

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=8)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=8)

    return train_loader, val_loader

# Model eğitimi fonksiyonu
def train_model(model, train_loader, val_loader, device, epochs=3):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    criterion = CrossEntropyLoss()
    dropout = Dropout(0.3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, inputs['labels'])
            loss = dropout(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Training loss: {total_loss / len(train_loader)}")

        model.eval()
        total_eval_accuracy = 0
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch[2]).cpu().numpy().mean() * 100
            total_eval_accuracy += accuracy
        print(f"Validation Accuracy: {total_eval_accuracy / len(val_loader)}")

def save_model_and_tokenizer(model, tokenizer, output_dir):
    # Eğer çıkış dizini yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_model_and_tokenizer(output_dir):
    print(f"Loading model from {output_dir}")
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    return model, tokenizer

def main():
    repo_name = 'facebook/react'  # Veri seti içeren repo
    auth_token = 'github_token'  # GitHub token girilmesi lazim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    commits = fetch_commits(repo_name, auth_token, max_pages=5)
    if commits:
        code_changes = extract_code_changes(commits, auth_token)
        if code_changes:
            train_loader, val_loader = prepare_dataset(code_changes, tokenizer)
            train_model(model, train_loader, val_loader, device)
            save_model_and_tokenizer(model, tokenizer, "model_output")
        else:
            print("No code changes extracted.")
    else:
        print("No commits fetched or data preparation failed.")

    # Kaydedilmiş modeli ve tokenizer'ı yüklemek için:
    # model, tokenizer = load_model_and_tokenizer("model_output")

if __name__ == '__main__':
    main()
