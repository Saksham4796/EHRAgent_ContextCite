import re
from context_cite import ContextCiter
import traceback

def split_sents(text):
    sentences = re.split(r'[.\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

class ContextCiteWrapper:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def score_memory_overall(self, query, response, recs):
        print(f"Scoring {len(recs)} memory records...")
        blocks = []
        for idx, r in enumerate(recs):
            task = r.get("question", "")
            reasoning = r.get("knowledge", "")
            code = r.get("code", "")

            combined_text = " ".join([task, reasoning, code])

            sentences = split_sents(combined_text)
            tagged = "\n".join(f"[[REC:{idx}]] {s}" for s in sentences)
            blocks.append(tagged)

        context = "\n\n".join(blocks)

        cc = ContextCiter.from_pretrained(
            self.model_name,
            context,
            query,
            device=self.device
        )

        _, prompt = cc._get_prompt_ids(return_prompt=True)
        cc._cache["output"] = prompt + response

        print(f"[ContextCite] ContextCiter initialized, computing attributions...")

        try:
            attributions = cc.get_attributions(
                start_idx=0,
                end_idx=len(response),
                as_dataframe=False,
                verbose=False
            )

            print(f"[ContextCite] Got {len(attributions)} attributions scores")
            sources = cc.sources

            rec_scores = {}
            for idx , (source, score) in enumerate(zip(sources, attributions)):

                match_id = re.search(r'\[\[REC:(\d+)\]\]', source)
                if match_id:
                    rec_id = int(match_id.group(1))
                    if rec_id not in rec_scores:
                        rec_scores[rec_id] = 0.0
                    rec_scores[rec_id] += float(score)

            print(f"[ContextCite] Computed scores: {rec_scores}")

            ranked = [(rec_scores[i], recs[i]) for i in range(len(recs))]
            ranked.sort(key=lambda x: x[0], reverse=True)

            return ranked
        
        except Exception as e:
            print(f"[ContextCite] Error during attribution computation: {e}")
            traceback.print_exc()

            print("[ContextCite] Falling back to uniform scores")
            return [(0.0, rec) for rec in recs]
        

    def score_memory_partition(self, query, response, recs):
        print(f"[ContextCite-Partition] Scoring {len(recs)} memory records...")
        
        blocks = []
        for idx, r in enumerate(recs):
            task = r.get("question", "")
            reasoning = r.get("knowledge", "")
            code = r.get("code", "")

            task_sentences = split_sents(task)
            reasoning_sentences = split_sents(reasoning)
            code_sentences = split_sents(code)

            task_tagged = "\n".join(f"[[REC:{idx}|TASK]] {s}" for s in task_sentences)
            reasoning_tagged = "\n".join(f"[[REC:{idx}|REASONING]] {s}" for s in reasoning_sentences)
            code_tagged = "\n".join(f"[[REC:{idx}|CODE]] {s}" for s in code_sentences)

            record_block = "\n".join([task_tagged, reasoning_tagged, code_tagged])
            blocks.append(record_block)

        context = "\n\n".join(blocks)

        cc = ContextCiter.from_pretrained(
            self.model_name,
            context,
            query,
            device = self.device
        )

        _, prompt = cc._get_prompt_ids(return_prompt=True)
        cc._cache["output"] = prompt + response

        print(f"[ContextCite-Partition] ContextCiter initialized, computing attributions...")

        try:
            attributions = cc.get_attributions(
                start_idx=0,
                end_idx= len(response),
                as_dataframe=False,
                verbose= False
            )

            print(f"[ContextCite-Partition] Got {len(attributions)} attributions scores")

            sources = cc.sources

            partition_scores = {}

            for idx, (source, score) in enumerate(zip(sources, attributions)):
                match_id = re.search(r'\[\[REC:(\d+)\|(\w+)\]\]', source)

                if match_id:
                    rec_id = int(match_id.group(1))
                    section = match_id.group(2)
                    score_val = float(score)

                    key = f"REC:{rec_id}|{section}"
                    if key not in partition_scores:
                        partition_scores[key] = 0.0
                    partition_scores[key] += score_val

            rec_max_scores = {}
            for key, total_score in partition_scores.items():
                rec_id = int(key.split('|')[0].split(':')[1])

                if rec_id not in rec_max_scores:
                    rec_max_scores[rec_id] = total_score

                else:
                    rec_max_scores[rec_id] = max(rec_max_scores[rec_id], total_score)

            ranked = [(rec_max_scores[i], recs[i]) for i in range(len(recs))]
            ranked.sort(key=lambda x:x[0], reverse=True)

            return ranked
        
        except Exception as e:
            print(f"[ContextCite-Partition] Error during attribution computation: {e}")
            traceback.print_exc()

            print("[ContextCite-Partition] Falling back to uniform scores")
            return [(0.0, rec) for rec in recs]