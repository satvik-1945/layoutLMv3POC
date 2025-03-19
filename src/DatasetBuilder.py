import os
import json
import datasets
from PIL import Image

class DatasetBuilder(datasets.GeneratorBasedBuilder):
    logger = datasets.logging.get_logger(__name__)

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(datasets.features.ClassLabel(
                        names=['B-ANSWER_RADIO', 'I-ANSWER_RADIO', 'E-ANSWER_RADIO', 'B-ANSWER_TEXT', 'I-ANSWER_TEXT',
                               'E-ANSWER_TEXT', 'B-QUESTION', 'I-QUESTION', 'E-QUESTION', 'B-TABLE',
                               'I-TABLE', 'E-TABLE',
                               'B-OTHERS', 'I-OTHERS', 'E-OTHERS'])),
                    "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            homepage="https://docex.probe42.in/",
        )

    def _split_generators(self, dl_manager):
        return_object = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(os.environ['SM_CHANNEL_TRAIN'], '')}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(os.environ['SM_CHANNEL_TEST'], '')}
            ),
        ]
        return return_object

    def get_line_bbox(self, bboxes):
        x = [bboxes[i][j] for i in range(len(bboxes)) for j in range(0, len(bboxes[i]), 2)]
        y = [bboxes[i][j] for i in range(len(bboxes)) for j in range(1, len(bboxes[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxes))]
        return bbox

    def normalize_bbox(self, bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

    def _generate_examples(self, filepath):
        self.logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            print(image_path)
            image_path = image_path.replace("json", "png")
            image, size = self.load_image(image_path)
            for item in data["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "others":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O  ")
                        cur_line_bboxes.append(self.normalize_bbox(w["bbox"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(self.normalize_bbox(words[0]["bbox"], size))
                    for w in words[1:-1]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(self.normalize_bbox(w["bbox"], size))
                    tokens.append(words[-1]["text"])
                    ner_tags.append("E-" + label.upper())
                    cur_line_bboxes.append(self.normalize_bbox(words[-1]["bbox"], size))
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, "image": image}