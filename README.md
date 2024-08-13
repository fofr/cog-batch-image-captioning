# cog-batch-image-captioning

A cog model for batch image captioning using various AI from OpenAI, Anthropic, and Google's Generative AI:

https://replicate.com/fofr/batch-image-captioning

## Features

- Process multiple images from a ZIP archive
- supports png, jpg, jpeg, webp
- Optional image resizing for more cost-effective captioning
- Customizable caption prefixes and suffixes
- Support for multiple AI models:
  - OpenAI: GPT-4 and variants
  - Anthropic: Claude-3.5, Claude-3 variants
  - Google: Gemini-1.5 variants
- Flexible system and message prompts
- Error handling and retry mechanism
- Output as a ZIP file containing captions that match image filenames as well as a CSV summary
