import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    # (we know nothing about the parents)
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # initialize probability variable
    probability = 1.0

    # loop over every person in people
    for person in people:

        # initialize individual probability
        individual_probability = 1.0

        # initialize parents
        father = people[person]["father"]
        mother = people[person]["mother"]

        parents = [father, mother]

        # if we want to calculate prob that the person has two genes
        if person in two_genes:

            # if that person has father and mother
            if father != None and mother != None:

                # the person needs to get a gene from each parent
                for parent in parents:

                    # if the parent has two genes
                    if parent in two_genes:
                        individual_probability *= (1 - PROBS["mutation"])

                    # if parent has one gene
                    elif parent in one_gene:
                        individual_probability *= 0.5

                    # if parent has no genes
                    else:
                        individual_probability *= PROBS["mutation"]

            # if person has no parents
            else:
                individual_probability *= PROBS["gene"][2]
            
            # if we want to calculate if that person has trait
            if person in have_trait:
                individual_probability *= PROBS["trait"][2][True]

            # if we want probability that the person doesn't have trait
            else:
                individual_probability *= PROBS["trait"][2][False]

        # if we want to calculate the prob that the person has one gene
        elif person in one_gene:

            # the person has parents
            if father != None and mother != None:

                # prob the father passes on gene but mother doesn't
                father_not_mother = 1.0

                # prob mother passes on gene but father doesn't
                mother_not_father = 1.0

                # father has two genes
                if father in two_genes:
                    father_not_mother *= (1 - PROBS["mutation"])
                    mother_not_father *= PROBS["mutation"]

                # father has one gene
                elif father in one_gene:
                    father_not_mother *= 0.5
                    mother_not_father *= 0.5

                # father has no genes
                else:
                    father_not_mother *= PROBS["mutation"]
                    mother_not_father *= (1 - PROBS["mutation"])

                # mother has two genes
                if mother in two_genes:
                    father_not_mother *= PROBS["mutation"]
                    mother_not_father *= (1 - PROBS["mutation"])

                # mother has one gene
                elif mother in one_gene:
                    father_not_mother *= 0.5
                    mother_not_father *= 0.5

                # mother has no genes
                else:
                    father_not_mother *= (1 - PROBS["mutation"])
                    mother_not_father *= PROBS["mutation"]

                # update probability
                individual_probability *= (father_not_mother + mother_not_father)

            # person has no parents
            else:
                individual_probability *= PROBS["gene"][1]

            # if we want to know prob person has trait
            if person in have_trait:
                individual_probability *= PROBS["trait"][1][True]

            # if we want to know if that person doesn't have trait
            else:
                individual_probability *= PROBS["trait"][1][False]

        # person has no genes
        else:

            # if person has parents doesn't receive gene from any of the parents
            if father != None and mother != None:
                
                # check both parents
                for parent in parents:
                    
                    # if parent has two genes
                    if parent in two_genes:
                        individual_probability *= PROBS["mutation"]

                    # if parent has one gene
                    elif parent in one_gene:
                        individual_probability *= 0.5

                    # if parent has no genes
                    else:
                        individual_probability *= (1 - PROBS["mutation"])

            # if person has no parents
            else:
                individual_probability *= PROBS["gene"][0]

            # if we want to calculate prob having the trait
            if person in have_trait:
                individual_probability *= PROBS["trait"][0][True]

            # if we want to calculate prob not having the trait
            else:
                individual_probability *= PROBS["trait"][0][False]

        # update probability
        probability *= individual_probability

    return probability
                

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # copy probabilities dict to iterate over it
    probabilities_copy = probabilities.copy()

    # for each person in probabilities (copy)
    for person in probabilities_copy:

        # case 1 gene
        if person in one_gene:
            probabilities[person]["gene"][1] += p

        # case 2 genes
        elif person in two_genes:
            probabilities[person]["gene"][2] += p

        # case no genes
        else:
            probabilities[person]["gene"][0] += p

        # case have trait
        if person in have_trait:
            probabilities[person]["trait"][True] += p

        # case no trait
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    # initialize copy of probabilities to iterate
    probabilities_copy = probabilities.copy()

    # for each person
    for person in probabilities_copy:

        # initialize suplementary variable
        gene_norm = 0.0

        # add all the values in probabilities[person]["gene"]
        for number_of_genes in range(3):
            gene_norm += probabilities_copy[person]["gene"][number_of_genes]

        # normalize gene values
        for number_of_genes in range(3):
            probabilities[person]["gene"][number_of_genes] /= gene_norm

        # add trait values
        trait_norm = probabilities[person]["trait"][True] + probabilities[person]["trait"][False]

        # normalize trait values
        probabilities[person]["trait"][True] /= trait_norm
        probabilities[person]["trait"][False] /= trait_norm


if __name__ == "__main__":
    main()