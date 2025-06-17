from accelerate import Accelerator
import torch

from configuration import load_yaml_config, ProjectConfig
from model.KG_LFM_arch import KG_LFM, KG_LFMConfig, set_KGLM_model_args

from utils.Dataloaders.pretrain_data import create_dataloader

from torch.optim import AdamW


def pretrain(config: ProjectConfig):
    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

    model_config = KG_LFMConfig.from_pretrained(
        config.model.llm_model_name,
        trust_remote_code=True,
    )
    model_config = set_KGLM_model_args(
        model_config, 
        config.model
    )
    
    model : KG_LFM = KG_LFM(model_config)
    
    # Create dataloaders
    train_dataloader, val_dataloader, _ = create_dataloader(
        config.dataset, config.pretrain_data, 
        model.tokenizer, "all"
    )
    
    # Prepare model and dataloaders with accelerator
    optimizer = AdamW(
        model.kg_encoder.parameters(),
        lr=config.model.kg_encoder_lr if config.model.kg_encoder_lr else 1e-4
    )
    
    train_dataloader, val_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, val_dataloader, model, optimizer
    )
    
    model.correct_train()
    
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/{10}")
        
        for step, batch in enumerate(train_dataloader):
            # breakpoint()
            outputs = model(
                input_ids=batch['input_ids'],
                graphs=batch['graphs'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids'],
                use_cache=True,
            )
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if step % config.pretrain_data.logging_steps == 0:
                print(f"Step {step}, Loss: {loss.item()}")
        
        # Validation step
        model.eval()
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_outputs = model(
                    input_ids=val_batch['input_ids'],
                    graphs=val_batch['graphs'],
                    attention_mask=val_batch['attention_mask'],
                    labels=val_batch['input_ids'],
                    use_cache=True,
                )
                val_loss = val_outputs.loss
                print(f"Validation Loss: {val_loss.item()}")
        
        model.correct_train()
        
    print("Pretraining complete.")
    
    # save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    unwrapped_model.save_pretrained(config.pretrain_data.output_dir, save_function=accelerator.save)
    print(f"Model saved to {config.pretrain_data.output_dir}")
    
    
def main():
    config = load_yaml_config("configlite.yaml")
    pretrain(config)
    
if __name__ == "__main__":
    main()