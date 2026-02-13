# Copyright 2026 pairsys.ai (DBA Goodmem.ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared pytest fixtures for integration tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def mock_receipt_pdf() -> bytes:
    """Generate a mock receipt PDF (Acme Corp -> GoodMind Inc.) and return raw bytes.

    The receipt contains specific addresses, line items, and a total that
    integration tests can verify via semantic retrieval.
    
    The PDF is also saved to mock_receipt.pdf in the repo root for visual inspection.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    import os

    NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}  # replaces ln=True

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # -- Header / Title --------------------------------------------------------
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "RECEIPT", align="C", **NL)
    pdf.ln(6)

    # -- From: Acme Corp -------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "From:", **NL)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Acme Corp", **NL)
    pdf.cell(0, 6, "123 Innovation Drive", **NL)
    pdf.cell(0, 6, "San Francisco, CA 94105", **NL)
    pdf.ln(4)

    # -- To: GoodMind Inc. -----------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "Bill To:", **NL)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "GoodMind Inc.", **NL)
    pdf.cell(0, 6, "456 Memory Lane", **NL)
    pdf.cell(0, 6, "Palo Alto, CA 94301", **NL)
    pdf.ln(4)

    # -- Date ------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(40, 7, "Date:")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "January 15, 2026", **NL)
    pdf.ln(6)

    # -- Line items table ------------------------------------------------------
    pdf.set_font("Helvetica", "B", 11)
    col_desc_w = 120
    col_amt_w = 50
    pdf.cell(col_desc_w, 8, "Description", border="B")
    pdf.cell(col_amt_w, 8, "Amount", border="B", align="R", **NL)

    items = [
        ("Cloud Computing Services", "$2,450.00"),
        ("Data Processing", "$1,275.50"),
        ("Technical Support", "$500.00"),
    ]
    pdf.set_font("Helvetica", "", 11)
    for desc, amt in items:
        pdf.cell(col_desc_w, 7, desc)
        pdf.cell(col_amt_w, 7, amt, align="R", **NL)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(col_desc_w, 8, "Total", border="T")
    pdf.cell(col_amt_w, 8, "$4,225.50", border="T", align="R", **NL)

    pdf_bytes = bytes(pdf.output())
    
    # Save to disk for visual inspection
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(repo_root, "mock_receipt.pdf")
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)
    
    return pdf_bytes
