from model_utils import load_checkpoint
import quant_summary
import os

def dump(trained_model_dir: str, trained_model_epoch: list, output_filename: str):
    model, bs = load_checkpoint(trained_model_dir, trained_model_epoch)

    if "train" not in trained_model_dir:
        fdir = os.path.join("train",trained_model_dir)
    else:
        fdir = trained_model_dir
    fname = os.path.join(fdir, output_filename)

    proposals = quant_summary.build_fp_proposal(model, w_bits=bs.w_bits, a_bits=bs.a_bits, o_bits=bs.o_bits, relu_unsigned=True)
    quant_summary.print_fp_proposal(proposals)
    quant_summary.save_fp_proposal_json(proposals, fname)
    print("Wrote proposal json file.")


if __name__ == "__main__":
    dump(trained_model_dir="trajectories_big_1_HD32_F32_B128_QBM_BML2_L4_BS16",
            trained_model_epoch=[150,],
            output_filename="fp_proposal_150.json")
    # dump(trained_model_dir="trajectories_big_1_HD32_F32_B128_QBM_BML2_L4_BS8",
    #         trained_model_epoch=[152,],
    #         output_filename="fp_proposal_152.json")