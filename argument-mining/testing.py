from argminer.data import ArgumentMiningDataset, TUDarmstadtProcessor
from argminer.evaluation import inference
from argminer.config import LABELS_MAP_DICT
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModelForTokenClassification, AutoTokenizer

# set path to data source
path = 'ArgumentAnnotatedEssay-2.0'

processor = TUDarmstadtProcessor(path)
processor = processor.preprocess()

# augmenter
def hello_world_augmenter(text):
    text = ['Hello'] + text.split() + ['World']
    text = ' '.join(text)
    return text

processor = processor.process('bieo', processors=[hello_world_augmenter]).postprocess()

df_dict = processor.get_tts(test_size=0.3)
df_train = df_dict['train'][['text', 'labels']]
df_test = df_dict['test'][['text', 'labels']]

df_label_map = LABELS_MAP_DICT['TUDarmstadt']['bieo']

max_length = 1024

# datasets
tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained('google/bigbird-roberta-base')
optimizer = Adam(model.parameters())

trainset = ArgumentMiningDataset(
    df_label_map, df_train, tokenizer, max_length
)
testset = ArgumentMiningDataset(
    df_label_map, df_train, tokenizer, max_length, is_train=False
)

train_loader = DataLoader(trainset)
test_loader = DataLoader(testset)

# sample training script (very simplistic, see run.py in cluster/cluster_setup/job_files for a full-fledged one)
epochs = 1
for epoch in range(epochs):
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        loss, outputs = model(
            labels=targets,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=False
        )

        # backward pass

        loss.backward()
        optimizer.step()

# run inference
df_metrics, df_scores = inference(model, test_loader)
