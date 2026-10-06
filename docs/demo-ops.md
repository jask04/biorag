# Demo recovery and availability

The demo uses the existing free Qdrant cluster and the CPU-basic Hugging
Face Space at https://huggingface.co/spaces/jask04/biorag.

## Recover a suspended cluster

1. Sign in to Qdrant Cloud with the existing account. Open `biorag` and
   select **Reactivate your cluster**. Keep the free tier; do not recreate
   the cluster or run `build_index.py` as part of recovery, because that
   script drops and rebuilds collections.
2. Wait for **HEALTHY**, then check both collections and their point counts.
   After the October 5, 2026 recovery, the BGE-small and PubMedBERT collections
   each contained 5,082 points, with 384- and 768-dimensional cosine vectors.
3. If GitHub disabled **Demo Availability**, re-enable the existing workflow:

   ```bash
   gh workflow enable demo-availability.yml --repo jask04/biorag
   gh workflow run demo-availability.yml --repo jask04/biorag --ref main
   ```

4. Open the Hugging Face demo and submit a question with the default hybrid
   retrieval and reranker. Confirm an answer with sources, then check the
   Benchmark tab. A successful HTTP response alone does not test the pipeline.

The Qdrant workflow check verifies that the demo collection exists, is
healthy and nonempty, and uses the expected 384-dimensional cosine vectors.
It only reads collection metadata; it never modifies points or credentials.
The Hugging Face HTTP check runs even if the Qdrant check fails.

## Free service limits

[Qdrant's free tier](https://qdrant.tech/documentation/cloud/create-cluster/)
suspends after one week of inactivity and deletes an unreactivated cluster
after four weeks of inactivity. The daily availability check supplies reads
while the GitHub workflow is active.

[GitHub scheduled workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)
can themselves be disabled after 60 days without repository activity. Their
scheduled runs do not guarantee permanent operation. Re-enable the workflow
if GitHub disables it and check the actual demo periodically. The workflow
does not create artificial commits or grant itself write access.

Shared Gemini quota is limited. A quota error can require waiting for the
quota reset or a visitor's own existing key; it is not a reason to recreate
the database or buy a paid plan.
