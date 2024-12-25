# Text to SQL

## QueryGPT

Uber used RAG and AI agents to build its in-house Text-to-SQL, saving 140,000 hours annually in query writing time.
Here’s how they built the system end-to-end:

The system is called QueryGPT and is built on top of multiple agents each handling a part of the pipeline.

1. First, the Intent Agent interprets user intent and figures out the domain workspace which is relevant to answer the
   question (e.g., Mobility, Billing, etc).
2. The Table Agent then selects suitable tables using an LLM, which users can also review and adjust.
3. Next, the Column Prune Agent filters out any unnecessary columns from large tables using RAG. This helps the schema
   fit within token limits.
4. Finally, QueryGPT uses Few-Shot Prompting with selected SQL samples and schemas to generate the query.

QueryGPT reduced query authoring time from 10 minutes to 3, saving over 140,000 hours annually!

https://www.uber.com/blog/query-gpt/

### From https://www.linkedin.com/in/juansequeda

About NL/Text-to-SQL implementations I’m seeing.

Many folks reach out to me about their NL/text-to-SQL apps they are building. I’ve also been reading and reviewing a lot
of blogs and scientific papers on the topic. The main observation is that folks are investing in the metadata,
semantics, knowledge, context…. but doing it adhoc. Many people ask me, “Do I really need to create a knowledge graph,
define ontologies, or build mappings? My system is already working pretty well. Why go through all that effort?”

Here’s the thing: You’re already doing the important work I talk about—defining the context.. For instance, when you add
descriptions about tables/columns, example values, example queries, or describe how things are joined—that is exactly
the type of semantics and knowledge you should be investing in. And it’s great to see people doing that! I’ve talked to
folks who say, “I’m getting 80% accuracy just by doing this work, and I didn’t need a knowledge graph or any of that
stuff.” That’s awesome—it shows the value of semantics and context.
But here’s the issue: the work you’re doing is ad hoc. You’re only doing it to make that specific text-to-SQL
application work. And worse, the semantics you’re codifying—those descriptions, examples, and metadata—often live in
places that aren’t well-managed. Maybe they’re embedded in code or prompts, or scattered across tools. That’s not
sustainable. That knowledge should be managed, governed, and reused. It shouldn’t just be for your text-to-SQL
application—it should benefit other applications and use cases too.
Sure, it works for now, but look at what you’re missing out on:

1. Governance – If you’re not managing that semantics and knowledge properly, you’re going to run into trouble when
   things change.
2. Scalability – What works for a proof-of-concept or pilot might not work long-term.
3. Reuse – The semantics you’re defining shouldn’t just help one application; they should create a foundation for other
   tools and use cases.

The point is this: semantics is an investment where 1 + 1 is greater than 2. For everyone already doing the work of
defining semantics, knowledge, context, and metadata—I applaud you. That’s the right first step. But take it further.
Manage it. Govern it. This should be part of a Business strategy (data should be reused to reduce cost and focus on
strategic initiatives) If you just cram that work into an application without thinking about reuse and scalability,
you’ll stay stuck in pilot and POC hell. Good luck with that.