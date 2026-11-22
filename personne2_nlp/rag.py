import glob
import sys
import os

import pandas as pd
from groq import Groq
import chromadb
import yaml
from sentence_transformers import SentenceTransformer

DOSSIER_DOCS = "documents"
CHEMIN_DB = "data/chroma"
NOM_COLLECTION = "maladies"
MODELE = "paraphrase-multilingual-MiniLM-L12-v2"

_modele = None


def get_modele():
    global _modele
    if _modele is None:
        _modele = SentenceTransformer(MODELE)
    return _modele


def charger_docs():
    """Lit tous les .md : en-tête YAML (entre ---) + texte."""
    docs = []
    for f in sorted(glob.glob(f"{DOSSIER_DOCS}/*.md")):
        brut = open(f, encoding="utf-8-sig").read()
        _, meta, texte = brut.split("---", 2)
        meta = yaml.safe_load(meta)
        meta["fichier"] = f
        meta["texte"] = texte.strip()
        docs.append(meta)
    return docs


def decouper(texte, max_mots=250, chevauchement=50):
    """Découpe par section (## ...). Le titre (# ...) est ajouté devant chaque chunk."""
    lignes = texte.splitlines()
    titre = ""
    if lignes and lignes[0].startswith("# "):
        titre = lignes[0]
        lignes = lignes[1:]

    sections, courant = [], []
    for ligne in lignes:
        if ligne.startswith("## ") and courant:
            sections.append("\n".join(courant))
            courant = []
        courant.append(ligne)
    if courant:
        sections.append("\n".join(courant))

    chunks = []
    for s in sections:
        s = s.strip()
        if not s:
            continue
        mots = s.split()
        if len(mots) <= max_mots:
            morceaux = [s]
        else:
            pas = max_mots - chevauchement
            morceaux = [" ".join(mots[i:i + max_mots]) for i in range(0, len(mots), pas)]
        for m in morceaux:
            chunks.append(f"{titre}\n{m}" if titre else m)
    return chunks


def construire_index():
    client = chromadb.PersistentClient(path=CHEMIN_DB)
    try:
        client.delete_collection(NOM_COLLECTION)
    except Exception:
        pass
    col = client.create_collection(NOM_COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, textes, metas = [], [], []
    for d in charger_docs():
        for i, c in enumerate(decouper(d["texte"])):
            ids.append(f'{d["fichier"]}#{i}')
            textes.append(c)
            metas.append({
                "fiche_kb": str(d["fiche_kb"]),
                "source": str(d["source"]),
                "url": str(d["url"]),
                "titre": str(d["titre"]),
            })

    embeddings = get_modele().encode(textes).tolist()
    col.add(ids=ids, documents=textes, embeddings=embeddings, metadatas=metas)
    print(f"{len(ids)} chunks indexés")


def chercher(fiche_kb, question, k=4):
    """Retourne les k chunks les plus proches, uniquement pour cette maladie."""
    col = chromadb.PersistentClient(path=CHEMIN_DB).get_collection(NOM_COLLECTION)
    emb = get_modele().encode(question).tolist()
    r = col.query(query_embeddings=[emb], n_results=k, where={"fiche_kb": fiche_kb})
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))

MODELE_LLM = "openai/gpt-oss-120b"

SYSTEME = """Tu es un assistant agricole. Tu réponds en français simple.
Règles :
- Utilise UNIQUEMENT le contexte fourni. N'ajoute aucune information extérieure.
- Commence par rappeler la maladie concernée.
- Attribue les affirmations à leur source (« selon l'université de ... ») et garde les réserves du contexte (« selon cette source », dates, nuances). N'écris jamais « à ce jour » ni une généralité absolue si le contexte ne le dit pas.
- Si le contexte ne contient pas la réponse, réponds exactement : "Je ne sais pas d'après mes sources."
- Reste court et précis."""


def nom_maladie(fiche_kb):
    df = pd.read_csv("data/correspondance.csv")
    ligne = df[df.fiche_kb == fiche_kb]
    if ligne.empty:
        return fiche_kb
    r = ligne.iloc[0]
    return f"{r.maladie_fr} ({r.plante})"


def answer(maladie, question, k=6):
    """Contrat avec la personne 3 : renvoie {"reponse": ..., "sources": [...]}."""
    passages = chercher(maladie, question, k)
    if not passages:
        return {
            "reponse": "Je n'ai pas encore de documents pour cette maladie.",
            "sources": [],
        }
    contexte = "\n\n---\n\n".join(
        f"[Source : {meta['source']}]\n{texte}" for texte, meta, _ in passages
    )
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    r = client.chat.completions.create(
        model=MODELE_LLM,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEME},
            {"role": "user", "content": (
                f"Maladie : {nom_maladie(maladie)}\n\n"
                f"Contexte :\n{contexte}\n\nQuestion : {question}"
            )},
        ],
    )
    sources = sorted({meta["url"] for _, meta, _ in passages})
    return {"reponse": r.choices[0].message.content, "sources": sources}


def classe_vers_fiche(classe_modele):
    df = pd.read_csv("data/correspondance.csv")
    ligne = df[df.classe_modele == classe_modele]
    return None if ligne.empty else ligne.iloc[0].fiche_kb

def repondre_depuis_classe(classe_modele, question):
    """Pont CV -> NLP : prend la sortie de predict()."""
    return answer(classe_vers_fiche(classe_modele), question)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "construire":
        construire_index()
    elif len(sys.argv) >= 4 and sys.argv[1] == "chercher":
        resultats = chercher(sys.argv[2], sys.argv[3])
        if not resultats:
            print("Aucun résultat pour cette fiche.")
        for texte, meta, dist in resultats:
            print(f"\n--- distance {dist:.3f} | {meta['source']}")
            print(texte[:300], "...")
    elif len(sys.argv) >= 4 and sys.argv[1] == "repondre":
        res = answer(sys.argv[2], sys.argv[3])
        print("\nRÉPONSE :\n" + res["reponse"])
        print("\nSOURCES :")
        for s in res["sources"]:
            print(" -", s)
    else:
        print('Usage : python rag.py construire | chercher <fiche> "<question>" | repondre <fiche> "<question>"')