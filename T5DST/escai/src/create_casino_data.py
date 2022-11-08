from pprint import pprint, pformat
import wandb
import json

from convokit import Corpus, download

from config import get_args


def create_casino_data(args):
    corpus = Corpus(
        filename=download(
            "casino-corpus", data_dir="/home/amogh.mannekote/myblue/.casino"
        )
    )
    for conv in corpus.iter_conversations():
        json.dump(conv.__dict__, open("test_casino_conv.txt", "w"))
        break


if __name__ == "__main__":
    # args = get_args()
    # config = vars(args)
    # config["dataset"] = "casino"
    # wandb.init(
    #     id=args.wandb_run_id,
    #     project="collaborative-dst",
    #     entity="msamogh",
    #     config=vars(args),
    # )

    create_casino_data(None)
