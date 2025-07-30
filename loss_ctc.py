
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    

class CTCLossCalculator1(nn.Module):
    def __init__(self, blank_class):
        super().__init__()
        self.blank_class = blank_class
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_class,
            zero_infinity=True,  # This is important
            reduction='none'
        )
    
    def forward(self, log_probs, targets, target_lengths, pred_lengths, _debug=False):
        """
        Args:
            ##  logits: Raw logits from model (B x T x C)
            log_probs: Log probabilities from model (B x T x C)
            targets: Target indices (B x S)
            target_lengths: Length of each target sequence (B,)
            pred_lengths: Length of each prediction sequence (B,)
        """
        # Apply log_softmax to get log probabilities if using raw logits as input
        #log_probs = F.log_softmax(logits, dim=2)

        
        # CTC loss expects (T x B x C)
        #log_probs = log_probs.permute(1, 0, 2)
        
        # Calculate CTC loss
        loss = self.ctc_loss(log_probs.transpose(0, 1), targets, pred_lengths, target_lengths)
        
        return loss.mean(), None# + self.blank_penalty * sil_penalty, None
        
        
        

class CTCLossCalculator_SILPenalty(nn.Module):

    def __init__(self, blank_class, sil_class=0, blank_penalty=0.1):
        super().__init__()
        self.blank_class = blank_class
        self.ctc_loss = nn.CTCLoss(blank=self.blank_class, 
                                  zero_infinity=True, 
                                  reduction='none')
        self.sil_class = sil_class
        self.blank_penalty = blank_penalty
    
    def forward(self, log_probs, targets, target_lengths, pred_lengths, _debug=False):
        batch_size = log_probs.size(0)
        max_pred_length = log_probs.size(1)
        
        # Permute for CTC loss
        log_probs = log_probs.permute(1, 0, 2)
        
        # Calculate base CTC loss
        loss = self.ctc_loss(log_probs, targets, pred_lengths, target_lengths)
        
        # Get blank probabilities [T, B]
        blank_probs = torch.exp(log_probs[:, :, self.blank_class])
        
        # Create position-dependent weights for blank and SIL penalties
        # Using cosine function: peaks in middle, lower at edges for blank
        # and opposite for SIL
        
        position_indices = torch.linspace(0, torch.pi, max_pred_length, device=log_probs.device)
        blank_weights = torch.cos(position_indices) * 0.4 + 0.6  # Range [0.2, 1.0]
        sil_weights = 1.0 - (torch.cos(position_indices) * 0.4 + 0.6)  # Range [1.0, 0.2]
        
        # Create SIL target mask with position-dependent weights
        target_mask = torch.zeros_like(blank_probs)  # Shape: [T, B]
        for b in range(batch_size):
            # Find positions of SIL in target sequence
            sil_positions = (targets[b, :target_lengths[b]] == self.sil_class).float()
            
            curr_pred_length = min(int(pred_lengths[b]), max_pred_length)
            ratio = float(curr_pred_length) / target_lengths[b].float()
            
            indices = torch.arange(0, curr_pred_length, device=log_probs.device).float()
            spread_indices = (indices / ratio).long()
            spread_indices = torch.clamp(spread_indices, max=target_lengths[b]-1)
            
            target_mask[:curr_pred_length, b] = sil_positions[spread_indices]
        
        # Apply position-dependent weights to penalties
        sil_penalty = (
            # Higher penalty at edges where we expect SIL
            0.5 * (blank_probs * target_mask * sil_weights.unsqueeze(1)).sum() +
            # Lower penalty in middle where we don't expect SIL
            0.01 * (blank_probs * (1 - target_mask) * (1 - sil_weights).unsqueeze(1)).sum()
        ) / batch_size
        
        # Add position-dependent blank penalty
        blank_penalty = (blank_probs * blank_weights.unsqueeze(1)).mean()
        
        total_loss = loss.mean() + (self.blank_penalty * (sil_penalty + 0.3 * blank_penalty))

        '''
        The main goal is to help the model learn when to use:

        SIL (silence) - should be at the start/end of words
        Blank - used by CTC for repeated phonemes
        Regular phonemes - the actual sounds in the word


        The loss has two main penalties:
        SIL Penalty:

        "Hey, if you see SIL in the target (start/end), but you're using blank instead, that's bad! Big penalty (0.5)"
        "If you use blank in the middle where there's no SIL, that's not as bad. Small penalty (0.01)"

        Blank Penalty:

        "In the middle of words, don't use too many blanks - we want clear phoneme predictions here"
        "At the edges, it's okay to use some blanks for transitions"
        '''
        
        return total_loss, None


        
    def forward_Feb16_old(self, log_probs, targets, target_lengths, pred_lengths):
        """
        Args:
            ##  logits: Raw logits from model (B x T x C)
            log_probs: Log probabilities from model (B x T x C)
            targets: Target indices (B x S)
            target_lengths: Length of each target sequence (B,)
            pred_lengths: Length of each prediction sequence (B,)
        """
        # Apply log_softmax to get log probabilities if using raw logits as input
        #log_probs = F.log_softmax(logits, dim=2)

        batch_size = log_probs.size(0)
        max_pred_length = log_probs.size(1)
        
        #print("log_probs.shape", log_probs.shape, targets.shape, pred_lengths.shape, target_lengths.shape)
        # CTC loss expects (T x B x C)
        log_probs = log_probs.permute(1, 0, 2)
        
        # Calculate CTC loss
        loss = self.ctc_loss(log_probs, targets, pred_lengths, target_lengths)
        
        '''
        blank_probs = torch.exp(log_probs[:, :, self.blank_class])
        # Add penalty for using blank instead of SIL
        sil_probs = torch.exp(log_probs[:, :, self.sil_class])
        blank_penalty = self.blank_penalty * (blank_probs - sil_probs).clamp(min=0).mean()
        '''

        # Calculate context-aware penalty
        # Get blank probabilities - note the shape after permute is [T, B, C]
        blank_probs = torch.exp(log_probs[:, :, self.blank_class])  # Shape: [T, B]
        
        # Create mask for positions where target is SIL
        target_mask = torch.zeros_like(blank_probs)  # Shape: [T, B]
        for b in range(batch_size):
            # Find positions of SIL in target sequence
            sil_positions = (targets[b, :target_lengths[b]] == self.sil_class).float()
            
            # Calculate the ratio while ensuring we don't exceed max_pred_length
            curr_pred_length = min(int(pred_lengths[b]), max_pred_length)
            ratio = float(curr_pred_length) / target_lengths[b].float()
            
            # Create indices for spreading, ensuring we don't exceed bounds
            indices = torch.arange(0, curr_pred_length, device=log_probs.device).float()
            spread_indices = (indices / ratio).long()
            spread_indices = torch.clamp(spread_indices, max=target_lengths[b]-1)
            
            # Apply mask safely - note we're assigning to columns now
            target_mask[:curr_pred_length, b] = sil_positions[spread_indices]
        
        # Calculate context-aware penalty
        sil_penalty = (
            # Higher penalty where we should have SIL
            0.5 * (blank_probs * target_mask).sum() +
            # Lower penalty elsewhere
            0.1 * (blank_probs * (1 - target_mask)).sum()
        ) / batch_size  # Normalize by batch size
        
        return loss.mean() + (self.blank_penalty * sil_penalty), None
    

class CTCLossCalculator3b(nn.Module): # a safer version of the previous one
    def __init__(self, blank_class):
        super().__init__()
        self.blank_class = blank_class
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_class,
            zero_infinity=True,  # This is important
            reduction='none'
        )
    
        

    def forward(self, log_probs, targets, target_lengths, pred_lengths):
        """
        Args:
            log_probs: Log probabilities from model (B x T x C)
            targets: Target indices (B x S)
            target_lengths: Length of each target sequence (B,)
            pred_lengths: Length of each prediction sequence (B,)
        """
        # Add value checks
        if torch.any(torch.isnan(log_probs)) or torch.any(torch.isinf(log_probs)):
            print("Warning: NaN or Inf in log_probs!")
            log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=0.0, neginf=-100.0)
        
        # Scale log_probs if they're too large
        max_val = log_probs.max().item()
        if max_val > 100:
            print(f"Warning: Large values in log_probs: {max_val}")
            log_probs = log_probs.clamp(min=-100.0, max=100.0)
        
        # Ensure inputs are in the correct range
        #log_probs = F.log_softmax(log_probs, dim=-1)  # Apply log_softmax again to ensure proper range
        
        # CTC loss expects (T x B x C)
        log_probs = log_probs.transpose(0, 1)
        
        # Calculate CTC loss with clipping
        try:
            loss = self.ctc_loss(log_probs, targets, pred_lengths, target_lengths)
            
            # Clip extremely large loss values
            loss = torch.clamp(loss, max=100.0)
            
            # Check loss values
            mean_loss = loss.mean()
            if mean_loss > 100 or torch.isnan(mean_loss) or torch.isinf(mean_loss):
                print(f"Warning: Loss issue - mean: {mean_loss.item()}, min: {loss.min().item()}, max: {loss.max().item()}")
                mean_loss = torch.clamp(mean_loss, max=100.0)
            
            return mean_loss, None
            
        except Exception as e:
            print(f"Error in CTC loss: {str(e)}")
            print(f"log_probs range: {log_probs.min().item()} to {log_probs.max().item()}")
            print(f"targets range: {targets.min().item()} to {targets.max().item()}")
            raise




if __name__ == "__main__":
    #test the model:
    torch.random.manual_seed(0)
    original_total_classes = 50
    blank_class = original_total_classes
    total_classes = original_total_classes + 1
    loss_fn = CTCLossCalculator_SILPenalty(blank_class=blank_class)
    print(loss_fn)
    
    #test the layer size calculator:
    batch_size = 5
    seq_len = 20
    output = torch.randn(batch_size, seq_len, total_classes)
    targets = torch.randint(0, original_total_classes, (batch_size, seq_len))
    target_lengths = torch.randint(int(seq_len/2), seq_len, (batch_size,))
    
    pred_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long)
    
    print("Batch outputs size:", output.shape)
    print("Batch targets size:", targets.shape)
    print("Batch target_lengths size:", target_lengths.shape)
    print("Target lengths:", target_lengths)

    preds = F.log_softmax(output, dim=2)

    loss_value = loss_fn(preds, targets, target_lengths, pred_lengths)
    print("loss_value", loss_value)