import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from PIL import Image
from torchvision import transforms
import random
import matplotlib.pyplot as plt


# 1. Load Processor and Model
model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(model_id)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)





lora_config = LoraConfig(
    r=8,            
    lora_alpha=16,  
    lora_dropout=0.15,
    target_modules=[
        # --- Language Decoder ---
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # --- Vision Encoder ---
        "out_proj", "fc1", "fc2"
    ],
    bias="none",
    task_type="CAUSAL_LM"
)




augmentation_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
])

def augment_image(pil_image: Image.Image) -> Image.Image:
    return augmentation_pipeline(pil_image)




def build_messages(conversations, pil_image):
    """
    Converts conversation turns into the VLM chat format.
    Image is attached as a placeholder to the first user message only.
    """
    messages = []
    for i, turn in enumerate(conversations):
        if turn["from"] == "human":
            content = []
            if i == 0:
                content.append({"type": "image"})  # image placeholder
            content.append({"type": "text", "text": turn["value"]})
            messages.append({"role": "user", "content": content})

        elif turn["from"] == "gpt":
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": turn["value"]}]
            })
    return messages




def format_train(example):
    """For training: applies random augmentation to the image."""
    pil_image = Image.open(example["image"]).convert("RGB")
    pil_image = augment_image(pil_image)
    messages = build_messages(example["conversations"], pil_image)
    return {"messages": messages, "images": [pil_image]}

def format_eval(example):
    """For evaluation: clean image, no augmentation."""
    pil_image = Image.open(example["image"]).convert("RGB")
    messages = build_messages(example["conversations"], pil_image)
    return {"messages": messages, "images": [pil_image]}



dataset_path = "powerline_vqa_dataset_insplad_enh.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")

total_size = len(dataset)
train_size = int(0.9 * total_size)

print(f"Total conversations : {total_size}")
print(f"Train (original)    : {train_size}")
print(f"Eval                : {total_size - train_size}")

raw_train = dataset.select(range(train_size))
raw_eval  = dataset.select(range(train_size, total_size))




def visualize_augmentations(dataset, num_samples=3, num_augs_per_image=4):
    """
    Plots the original image next to several augmented versions of itself.
    """
    print("Generating augmentation samples for review...")

    # Create a grid of plots
    fig, axes = plt.subplots(num_samples, num_augs_per_image + 1, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        # 1. Load the original image
        example = dataset[i]
        original_image = Image.open(example["image"]).convert("RGB")

        # Plot Original
        ax_orig = axes[i, 0] if num_samples > 1 else axes[0]
        ax_orig.imshow(original_image)
        ax_orig.set_title("Original Image")
        ax_orig.axis("off")

        # Plot Augmentations
        for j in range(num_augs_per_image):
            # Apply your augmentation pipeline
            augmented_image = augment_image(original_image)

            ax_aug = axes[i, j + 1] if num_samples > 1 else axes[j + 1]
            ax_aug.imshow(augmented_image)
            ax_aug.set_title(f"Augmentation {j + 1}")
            ax_aug.axis("off")

    plt.tight_layout()
    # plt.savefig("augmentation_samples.png", bbox_inches='tight')
    # print("Saved 'augmentation_samples.png'. Please review this file!")
    plt.show()


visualize_augmentations(raw_train, num_samples=3, num_augs_per_image=4)


NUM_AUGMENTATIONS = 5

repeated_indices = list(range(train_size)) * NUM_AUGMENTATIONS
augmented_raw_train = raw_train.select(repeated_indices)
augmented_raw_train = augmented_raw_train.shuffle(seed=42)

print(f"Train after {NUM_AUGMENTATIONS}x augmentation: {len(augmented_raw_train)}")



print("Formatting train dataset (augmented)...")
train_dataset = augmented_raw_train.map(
    format_train,
    remove_columns=["image", "conversations", "id"]
)

print("Formatting eval dataset (clean)...")
eval_dataset = raw_eval.map(
    format_eval,
    remove_columns=["image", "conversations", "id"]
)


def vlm_data_collator(examples):
    texts = []
    images = []

    for example in examples:
        text = processor.apply_chat_template(example["messages"], add_generation_prompt=False)
        texts.append(text)
        images.append(example["images"][0])

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()

    # 1. Grab our special tokens
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    eou_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")

    # We find the token sequence for "Assistant:" so we don't train the model to repeat it
    assistant_header_ids = processor.tokenizer.encode("Assistant:", add_special_tokens=False)
    header_len = len(assistant_header_ids)

    # 2. Iterate through each conversation in the batch
    for i in range(len(labels)):
        seq = labels[i]

        # Create a blank slate of -100 (mask everything by default)
        new_label_seq = torch.full_like(seq, -100)

        # Find all <end_of_utterance> tokens to divide the conversation into "turns"
        eou_indices = (seq == eou_token_id).nonzero(as_tuple=True)[0].tolist()

        start_idx = 0
        is_assistant_turn = False # Turn 0 is always the User asking the question

        for eou_idx in eou_indices:
            end_idx = eou_idx + 1 # Include the <end_of_utterance> token so the model learns to stop

            # If it is the assistant's turn, we want to UNMASK it so it gets graded
            if is_assistant_turn:
                chunk = seq[start_idx:end_idx].tolist()

                # Find exactly where "Assistant:" ends so we only grade the actual answer text
                answer_start_offset = 0
                for j in range(len(chunk) - header_len + 1):
                    if chunk[j : j + header_len] == assistant_header_ids:
                        answer_start_offset = j + header_len
                        break

                actual_answer_start = start_idx + answer_start_offset

                # Unmask the actual answer!
                new_label_seq[actual_answer_start:end_idx] = seq[actual_answer_start:end_idx]

            # Move to the next turn and swap roles
            start_idx = end_idx
            is_assistant_turn = not is_assistant_turn

        # Handle trailing text if the sequence was truncated due to max_length
        if start_idx < len(seq) and is_assistant_turn:
            chunk = seq[start_idx:].tolist()
            answer_start_offset = 0
            for j in range(len(chunk) - header_len + 1):
                if chunk[j : j + header_len] == assistant_header_ids:
                    answer_start_offset = j + header_len
                    break
            actual_answer_start = start_idx + answer_start_offset
            new_label_seq[actual_answer_start:] = seq[actual_answer_start:]

        # Re-apply padding and image masking just 
        new_label_seq[seq == pad_token_id] = -100
        new_label_seq[seq == image_token_id] = -100

        labels[i] = new_label_seq

    batch["labels"] = labels
    return batch



def verify_collator(dataset, collator, processor, num_samples=2):
    print("=== SANITY CHECK: DATA COLLATOR ===")

    # 1. Grab a few samples from your formatted dataset
    samples = [dataset[i] for i in range(num_samples)]

    # 2. Pass them through your custom collator
    batch = collator(samples)

    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Pixel Values shape: {batch['pixel_values'].shape}\n") # If model uses pixel_values

    # 3. Decode and inspect the first example in the batch
    print("--- Example 0 Inspection ---")

    # Decode what the model READS (input_ids)
    # We replace -100 with the pad token ID just so the decoder doesn't crash
    input_ids = batch["input_ids"][0].clone()
    read_text = processor.decode(input_ids, skip_special_tokens=False)
    print("WHAT THE MODEL READS (`input_ids`):\n", read_text)
    print("-" * 40)

    # Decode what the model is GRADED ON (labels)
    labels = batch["labels"][0].clone()
    # To decode, we filter out all the -100 values
    graded_ids = labels[labels != -100]
    graded_text = processor.decode(graded_ids, skip_special_tokens=False)
    print("WHAT THE MODEL IS GRADED ON (`labels` != -100):\n", graded_text)
    print("===================================\n")

# Run the test
verify_collator(train_dataset, vlm_data_collator, processor)



training_args = SFTConfig(
    output_dir="smolvlm-powerline-adapter",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_steps=1000,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False 
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=training_args,
    data_collator=vlm_data_collator, 
    processing_class=processor,
)

print("\nTrainable parameters:")
trainer.model.print_trainable_parameters()

print("\nStarting training...")
trainer.train()

trainer.save_model("smolvlm-powerline-adapter")
print("Done. Best model saved to smolvlm-powerline-adapter/")



