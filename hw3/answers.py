r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 500
    hypers["seq_len"] = 10
    hypers["h_dim"] = 256
    hypers["n_layers"] = 4
    hypers["dropout"] = 0.1
    hypers["learn_rate"] = 0.0005
    hypers["lr_sched_factor"] = 0.1
    hypers["lr_sched_patience"] = 2
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Act I."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
splitting the corpus into sequences instead of the whole text is trainnig for number of resons. during the backprop, the problem of vamising gradients becouse the network si too long (too deep) for the whole text, cousing the gradients to minimizs so much then nearly vanish in the end. also RAM memory limit - whole text means many hidden states to calculate the gradient, courpus is like traning on smallet batch sizes in CNN. also updating the weights after small epochs and not after going thought the text a single time. 
"""

part1_q2 = r"""
The generated text is longer than the sequence length becouse there is difference between trainning on the sequence and generating a noval text. even in the trainnig part - the batch size (the number of sentences) is kept identical but the number of chars in every sentance is not identical and variating. in the trainig the hidden state is gated to prevent vanishing gradients but on testing (generating) the hidden state can be forwarded forever. 
"""

part1_q3 = r"""
When trainning on RNNs we do not shuffle the order of batches becouse we want to also learn the logic of the text, for that, keeping the order of the batches is crutial. shuffeling can theoreticly creat a situations where learning the first a sentece from the middle of the text and then a sentence from the end of the text, loosing context, logic and connections between the sentances. 
"""

part1_q4 = r"""
1. temperature of 1 means the probabilities are raw - no scalling added. We lower the temperature in RNNS in the final layer to make the model more confident and coherent in its pradictions. to get rid by filter out values with low probabilites (that can be considers as noise). 

2. When temperature is very high and larger than 1 - this means the distribution of the values becomes similar to unifrom distribution (as T goes to inf.). as T->inf then {z_i/T}->0 and then exp(z_i/T)~exp(0)=1 for every z_i value. this means the text is generated randomly - no value has an advantage over other values. 

3. When the temperature is very low and smaller than 1 - this means the distribution is more pointy (like a deltha function). the model is deterministic and the generated text is less creative and less random (less noisy). this is becouse T->0 then {z_i/T}->inf. and then exp(z_i/T)~exp(inf.)~inf for the largest z_i value, this one have the higher chages of being choosen by the model as the mext char.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 32
    hypers["h_dim"] = 1024
    hypers["z_dim"] = 256
    hypers["x_sigma2"] = 0.0001
    hypers["learn_rate"] = 0.0001
    hypers["betas"] = (0.85, 0.9999)
    # ========================
    return hypers


part2_q1 = r"""
{sigma ** 2} hyperparameter is the assumed variance of the of the parametric liklihood function of the decoder, which is assumed to be normal gaussian distributed. 
the eqution we were given shows that the VAE_loss is related to {1/sigma ** 2}. this means that low {sigma ** 2} causes higher reconstraced loss (dominant over KL loss) meaning the model is a good reconstructor, the images look more similar to the train data and less creative. 
higher {sigma ** 2} means less dominant reconstruction loss (dominant over the reconstruction loss) meaning the model less reconstructs and more creative, the results looks less similar to the train data. 
"""

part2_q2 = r"""
1. the purpose of reconstruction loss - to control how much the generated data is similar to the original training data. if this loss is high, this means the generated data is distriduted similar to the training data.  
the purpose KL divergence loss - is similar to regularization, doces the latent distribution to be similar to a normal distributed prior. loss is higher the model is more creative and distributed less like the trainnig data and more like a normal distribution, a more simple prior distribution. 

2. KL loss prevents overfitting in the latent-space distribution. it forces the autocoder to center around mean zero and variance of around 1. without KL the coder will scatter the points in the latent space and will make the variance of each point small. this will creat a model that can reconstruct perfectly but can not generate new content.   

3. the benefit of this effect is a more continious latent space, meaning more continous generation ability of the model and no dead zones. mapping all of the points in the lattent space, not leaving a gap. such gap means the model will choose a value withoud context, meaning gibrish.  
"""

part2_q3 = r"""
start by maximizing the evidence for the VAE loss distribution, p(x) is the evidance we maximize, this is the truth we want the model to capture. we want the model to learn first the highest probabilites to generate an output that is somewat similr to the original data (starting point) and then to add to them and generate a more creative and specific result. 
for example, learn the stracture of a general face and them add specific facial featurs to generate a new face. 
"""

part2_q4 = r"""
use the log of the latent-space variance rather then directly calculating the mean and the variance. this is more mathematical reason. when working with sigma ** 2 that needs to be positive, when updating the gradients the update can take sigma ** 2 to a negative value, that is not possible as variacne is always possitive (power of 2). 
log of a positive value can be negative / 1 / positive, no matter where the gradient updating will take the log(var ** 2), the varianc can be calculated as exp(log(sigma ** 2)). 
also for numerial stability purposes and optimizing smaller values of the sigma ** 2 (near the zero). 
"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers['embed_dim'] = 128
    hypers['num_heads'] = 16
    hypers['num_layers'] = 4
    hypers['hidden_dim'] = 64
    hypers['window_size'] = 100
    hypers['droupout'] = 0.15
    hypers['lr'] = 0.0005
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
