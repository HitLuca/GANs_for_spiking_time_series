from image_generator import ImageGenerator

generator_model_filepath = '../generative_models/wgan_gp/outputs/final/packing/models/1000000/generator.h5'
losses_filepath = '../generative_models/wgan_gp/outputs/final/packing/losses.p'

# image_generator = ImageGenerator(generator_model_filepath)
# image_generator.generate_samples()
# image_generator.generate_latent_space()

image_generator = ImageGenerator(losses_filepath, losses_end=-1)
image_generator.generate_losses(mean_filter=True, mean_filter_size=1000)
