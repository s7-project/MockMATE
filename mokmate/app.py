
import streamlit as st

st.set_page_config(
    page_title="MockMATE",
    page_icon="🎯",
    layout="centered"
)

QUESTIONS = {
    "Python Basics": [
        {
            "question": "What is the difference between a list and a tuple?",
            "keywords": ["list", "tuple", "mutable", "immutable"],
            "answer": (
                "A list is mutable, meaning its elements can be changed. "
                "A tuple is immutable, meaning its elements cannot be "
                "changed after creation."
            ),
        },
        {
            "question": "What is the purpose of a Python dictionary?",
            "keywords": ["key", "value"],
            "answer": (
                "A dictionary stores data as key-value pairs. "
                "Keys must be unique."
            ),
        },
    ],
    "OOP": [
        {
            "question": "What is inheritance in Python?",
            "keywords": ["class", "parent", "child"],
            "answer": (
                "Inheritance allows a child class to reuse or extend "
                "attributes and methods from a parent class."
            ),
        }
    ],
    "Exception Handling": [
        {
            "question": "What is the purpose of try and except?",
            "keywords": ["try", "except", "exception"],
            "answer": (
                "The try block contains code that may raise an exception. "
                "The except block handles a matching exception."
            ),
        }
    ],
}


def evaluate_answer(answer, keywords):
    answer = answer.lower()
    matched = [word for word in keywords if word in answer]
    score = round(len(matched) / len(keywords) * 100)

    return score, matched


st.title("🎯 MockMATE")
st.subheader("AI Mock Interview — Python")
st.write("Practice Python interview questions and receive feedback.")

topic = st.selectbox("Choose an interview topic", list(QUESTIONS))

if (
    "topic" not in st.session_state
    or st.session_state.topic != topic
):
    st.session_state.topic = topic
    st.session_state.question_index = 0
    st.session_state.answer = ""
    st.session_state.result = None

question_list = QUESTIONS[topic]
index = st.session_state.question_index
current = question_list[index]

st.progress((index + 1) / len(question_list))
st.caption(f"Question {index + 1} of {len(question_list)}")

st.markdown("### Interview Question")
st.info(current["question"])

answer = st.text_area(
    "Your answer",
    key="answer",
    height=150,
    placeholder="Type your answer here..."
)

if st.button("Evaluate Answer", type="primary"):
    if not answer.strip():
        st.warning("Please enter your answer first.")
    else:
        score, matched = evaluate_answer(
            answer, current["keywords"]
        )
        st.session_state.result = {
            "score": score,
            "matched": matched,
            "reference": current["answer"],
            "keywords": current["keywords"],
        }

if st.session_state.result is not None:
    result = st.session_state.result

    st.markdown("### Your Feedback")
    st.metric("Keyword coverage (not an accuracy score)", f"{result['score']}%")

    if result["matched"]:
        st.write("**Concept terms detected:** " + ", ".join(result["matched"]))
    else:
        st.write("No target terms were detected.")

    missing = [
        word for word in result["keywords"]
        if word not in result["matched"]
    ]

    if missing:
        st.write("**Terms to review:** " + ", ".join(missing))

    st.markdown("**Reference answer**")
    st.write(result["reference"])

    st.caption(
        "This first version checks keywords only. It does not yet "
        "understand meaning or reliably judge correctness."
    )

    if index + 1 < len(question_list):
        if st.button("Next Question"):
            st.session_state.question_index += 1
            st.session_state.answer = ""
            st.session_state.result = None
            st.rerun()
    else:
        st.success("You completed this topic's questions!")
        if st.button("Restart Topic"):
            st.session_state.question_index = 0
            st.session_state.answer = ""
            st.session_state.result = None
            st.rerun()