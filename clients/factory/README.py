# Factory

This mflux-server client is a collection of scripts which can be used to mass-produce images based on mass-produced prompts. The nucleus of this production pipeline is a category selection which is first expanded into prompts and that large list of prompts is used to make images with mflux-server.

## Workflow

```
factory-categories.json
  |
  | prompt-factory.py
  |
  v
factory-prompts.jsonlist
  |
  | image-factory.py
  |
  v
images/
```

So basically you only need to run

```
python3 prompt-factory.py
python3 image-factory.py
```

... which should start a mass-production of images in the `images` folder.

The image client can attach the same ordered image inputs to every prompt:

```sh
python3.12 image-factory.py --server http://localhost:4030 --init-images object.png room.png
```

It uses `/api/info` to check image support and count before sending `init_images`
to `/api/generate`. Unsupported inputs stop the run rather than being omitted.

## Fine-Tuning

If you want images about topics of your own, you have two options to influence the image production:

- To get more of a different content add more catories to the file `factory-categories.json`.
- To get less of an existing content filter out those i.e. with a `grep -f unwanted-word factory-prompts.jsonlist > factory-prompts.jsonlist`


