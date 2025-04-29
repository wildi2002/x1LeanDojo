import torch
from transformers import AutoModel, AutoTokenizer

class Lean4Verification:
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            "internlm/internlm2_5-step-prover-critic", 
            device_map="cuda", 
            torch_dtype=torch.float16, 
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-step-prover-critic", trust_remote_code=True)

    def get_chat(self, problem, cot):
        return [{f"role": "user", "content": "{problem}"}, {"role": "assistant", "content": "{cot}"}]
    
    def compare(self, problem, cot1, cot2):
        print(self.get_chat(problem, cot1))
        return f"Score 1: \t {self.model.get_score(self.tokenizer, self.get_chat(problem, cot1))} \t\t Score 2: \t {self.model.get_score(self.tokenizer, self.get_chat(problem, cot2))}"


if __name__ == "__main__":
    lean = Lean4Verification()

    problem = "Show that the sum of two even numbers is always even."
    print(lean.compare(problem, ("""An even number can be written as 2 times a natural number. Let a = 2n and b = 2m for some natural numbers n and m. Then a + b = 2n + 2m. Factor the expression: 2n + 2m = 2(n + m). Since n + m is a natural number, a + b is divisible by 2, hence even."""), 
                                ("""Even numbers are defined as numbers that are divisible by 2. Assume a and b are even, so there exist integers k and l such that a = 2k and b = 2l. Then, a + b = 2k + 2l. This simplifies to a + b = 2(k + l). Therefore, a + b is divisible by 2, so it is even.""")))

    problem = "Prove that the product of two odd numbers is always odd."
    print(lean.compare(problem,
        ("""Let a = 2n + 1 and b = 2m + 1 be two odd numbers. Then a * b = (2n + 1)(2m + 1) = 4nm + 2n + 2m + 1 = 2(2nm + n + m) + 1, which is of the form 2k + 1, so it is odd."""),
        ("""Odd numbers can be written as 2k + 1. The product is (2k + 1)(2l + 1) = 4kl + 2k + 2l + 1 = 2(2kl + k + l) + 1. Since this is one more than an even number, the product is odd.""")))

    problem = "Prove that the square of an even number is even."
    print(lean.compare(problem,
        ("""Let a be an even number, so a = 2n for some integer n. Then a² = (2n)² = 4n² = 2(2n²), which is divisible by 2, hence even."""),
        ("""An even number can be expressed as 2k. Squaring gives (2k)² = 4k² = 2(2k²), which is clearly even since it's a multiple of 2.""")))

    problem = "Show that the sum of an odd number of consecutive integers is divisible by the number of terms."
    print(lean.compare(problem,
        ("""Let the sequence be centered at 0: -n, ..., 0, ..., n. The number of terms is 2n + 1, which is odd. Their sum is zero due to symmetry, and zero is divisible by any integer, including 2n + 1."""),
        ("""Let the numbers be a, a+1, ..., a+2n. The number of terms is 2n + 1. The sum is (2n + 1)(a + n), and since (2n + 1) divides the product, it divides the sum.""")))

    print()
    print("Erste COT korrekt.")
    print("_"*20)

    problem = "Prove that the square of any odd number is odd."
    print(lean.compare(problem,
        ("""Let a be an odd number, so a = 2n + 1. Then a² = (2n + 1)² = 4n² + 4n + 1 = 2(2n² + 2n) + 1, which is of the form 2k + 1, hence odd."""),  # ✅ korrekt
        ("""Odd numbers are numbers that end in 1, 3, 5, 7, or 9. Squaring these always gives a number that ends in 1, 3, 5, 7, or 9, so the result is odd."""  # ❌ falsch – basiert auf Ziffern, nicht Definition
    )))
    problem = "Show that the product of any number and 0 is 0."
    print(lean.compare(problem,
        ("""Let a be any number. Then a × 0 = 0 by the definition of multiplication with zero."""),  # ✅ korrekt
        ("""Zero times a number is the same as that number because multiplying means repeating addition."""  # ❌ falsch – beschreibt Addition falsch
    )))


    print()
    print("Zweite COT korrekt.")
    print("_"*20)

    problem = "Prove that 0 is an even number."
    print(lean.compare(problem,
        ("""0 is not even because it is not positive and even numbers must be greater than 0."""),  # ❌ falsch – falsche Definition
        ("""A number is even if it is divisible by 2. Since 0 ÷ 2 = 0 with no remainder, 0 is even."""  # ✅ korrekt
    )))
    problem = "Prove that the sum of interior angles in a triangle is 180 degrees (Euclidean geometry)."
    print(lean.compare(problem,
        ("""Triangles have three sides, and each side contributes 60 degrees, making 180 degrees."""),  # ❌ falsch – keine Begründung, falsche Annahme
        ("""Draw a line parallel to one side of the triangle through the opposite vertex. Alternate angles formed equal the triangle's angles and form a straight line, so their sum is 180 degrees."""  # ✅ korrekt
    )))


    print()
    print("Zweite COT korrekt.")
    print("_"*20)

    problem = "Is the sum of a rational and an irrational number always irrational?"
    print(lean.compare(problem,
        ("""Yes, because adding any two numbers always gives a rational number."""),  # ❌ falsch
        ("""No, because the irrational part cancels out the rational one."""  # ❌ falsch – unklar und missverständlich
    )))