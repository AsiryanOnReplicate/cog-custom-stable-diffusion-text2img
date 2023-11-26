# Stable Diffusion

This is an implementation of a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run

**text2img** predictions:

    cog predict -i prompt="your prompt"

