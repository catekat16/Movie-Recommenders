import contextlib
import textwrap
import traceback

import streamlit as st
from streamlit import source_util


@contextlib.contextmanager
def maybe_echo():
    if not st.checkbox("Show Code"):
        yield
        return

    code = st.empty()
    try:
        frame = traceback.extract_stack()[-3]
        filename, start_line = frame.filename, frame.lineno

        yield

        frame = traceback.extract_stack()[-3]
        end_line = frame.lineno
        lines_to_display = []
        with source_util.open_python_file(filename) as source_file:
            source_lines = source_file.readlines()
            lines_to_display.extend(source_lines[start_line:end_line])
            initial_spaces = st._SPACES_RE.match(lines_to_display[0]).end()
            for line in source_lines[end_line:]:
                indentation = st._SPACES_RE.match(line).end()
                # The != 1 is because we want to allow '\n' between sections.
                if indentation != 1 and indentation < initial_spaces:
                    break
                lines_to_display.append(line)
        lines_to_display = textwrap.dedent("".join(lines_to_display))

        code.code(lines_to_display, language='python')

    except FileNotFoundError as err:
        code.warning("Unable to display code. %s" % err)


def format_movie(movie_string):
    movie_string = movie_string[:-7] # get rid of the year
    #print("before split: ", movie_string)
    movie_string = movie_string.split(',')
    #print("after split: ", movie_string)

    if len(movie_string) > 0:
        movie_string = movie_string[-1] + ' ' + ''.join(movie_string[:-1]) # want the stuff after , to be at the front of the string
        #print("new string: ", movie_string)

    return movie_string

#def maybe_echo1(lines_to_display):
#    if not st.checkbox("Show Code"):
#        yield
#        return

#    st.code(lines_to_display, language='python')

