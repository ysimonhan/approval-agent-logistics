from __future__ import annotations

import datetime
import uuid

import pandas as pd
import streamlit as st

from logistics_approval_agent.config import Settings, get_settings
from logistics_approval_agent.services.decision_engine import (
    determine_ticket_status,
    process_ticket_with_ai,
)
from logistics_approval_agent.services.llm_cohere import (
    analyze_sop_with_cohere,
    chat_with_ai_agent,
)
from logistics_approval_agent.services.ocr_mistral import process_sop_with_mistral_ocr


def generate_ticket_id() -> str:
    return str(uuid.uuid4())[:8]


def get_current_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_action(ticket_id, action, details=None, user="System") -> None:
    log_entry = {
        "timestamp": get_current_timestamp(),
        "ticket_id": ticket_id,
        "action": action,
        "user": user,
        "details": details if details is not None else {},
    }
    st.session_state.logs.append(log_entry)
    print(f"LOG: {log_entry}")


def init_session_state() -> None:
    if "tickets" not in st.session_state:
        st.session_state.tickets = []
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = []
    if "age_cost_thresholds" not in st.session_state:
        st.session_state.age_cost_thresholds = {
            "New (0-2 years)": 1000,
            "Medium (3-5 years)": 700,
            "Old (6-8 years)": 400,
            "Very Old (>8 years)": 200,
        }
    if "repair_codes_needing_images" not in st.session_state:
        st.session_state.repair_codes_needing_images = {
            "DMG001": "Severe Dent Repair",
            "CRK003": "Frame Crack Assessment",
            "RST005": "Surface Rust Treatment",
            "FLR001": "Floor Replacement",
        }
    if "current_repairs_for_new_ticket" not in st.session_state:
        st.session_state.current_repairs_for_new_ticket = []
    if "current_media_for_new_ticket_with_assoc" not in st.session_state:
        st.session_state.current_media_for_new_ticket_with_assoc = []


def run_ai_pipeline(ticket: dict, settings: Settings, use_cohere_aya: bool) -> dict:
    return process_ticket_with_ai(
        ticket_data=ticket,
        age_cost_thresholds=st.session_state.age_cost_thresholds,
        repair_codes_needing_images=st.session_state.repair_codes_needing_images,
        use_cohere_aya=use_cohere_aya,
        settings=settings,
        info_callback=st.sidebar.info,
        error_callback=st.error,
        success_callback=st.sidebar.success,
        warning_callback=st.warning,
    )


def apply_ai_result(ticket: dict, ai_result: dict, *, source: str) -> str:
    ticket.update(
        {
            "ai_decision": ai_result["decision"],
            "ai_confidence": ai_result.get("confidence_score", 0.0),
            "ai_reasoning": ai_result["reasoning"],
            "ai_agent_type": ai_result.get("ai_agent_type", "Unknown AI"),
            "ai_processed_date": get_current_timestamp(),
            "ai_missing_data_request": ai_result.get("missing_data_request"),
        }
    )

    status = determine_ticket_status(ai_result)
    ticket["status"] = status

    if status == "Additional Data Requested":
        message = (
            f"Additional data required: {ticket['ai_missing_data_request']}. "
            f"Reasoning: {ticket['ai_reasoning']}"
        )
        log_msg_action = f"Additional Data Requested by AI ({source})"
    elif status == "AI Approved":
        message = f"Ticket Approved. Reasoning: {ticket['ai_reasoning']}"
        log_msg_action = f"AI Approved ({source})"
    elif status == "AI Disapproved":
        message = f"Ticket Disapproved. Reasoning: {ticket['ai_reasoning']}"
        log_msg_action = f"AI Disapproved ({source})"
    else:
        message = f"This ticket requires manual review. Reasoning: {ticket['ai_reasoning']}"
        log_msg_action = f"Flagged for Manual Review by AI ({source})"

    ticket.setdefault("ai_chat", []).append(
        {
            "timestamp": get_current_timestamp(),
            "sender": "AI Agent",
            "message": message,
        }
    )
    return log_msg_action


def add_sample_tickets_if_needed(settings: Settings) -> None:
    if st.session_state.tickets:
        return

    sample_tickets_data_raw = [
        {
            "container_id": generate_ticket_id(),
            "company": "Maersk",
            "total_cost_estimate": 800,
            "container_age": 1,
            "repairs": [
                {"code": "DNT001", "description": "Minor dent"},
                {"code": "SCT002", "description": "Scratch"},
            ],
            "media_simulated": [
                {
                    "filename": "img1_for_DNT001.jpg",
                    "type": "image",
                    "repair_code_association": "DNT001",
                }
            ],
            "other_notes": "Standard wear.",
        },
        {
            "container_id": generate_ticket_id(),
            "company": "MSC",
            "total_cost_estimate": 1200,
            "container_age": 4,
            "repairs": [
                {"code": "DMG001", "description": "Structural damage"},
                {"code": "FLR001", "description": "Floor replace"},
            ],
            "media_simulated": [],
            "other_notes": "Urgent.",
        },
        {
            "ticket_id": generate_ticket_id(),
            "container_id": "CON003",
            "company": "Cosco",
            "total_cost_estimate": 300,
            "container_age": 7,
            "repairs": [{"code": "RST005", "description": "Surface rust treatment"}],
            "media": [{"filename": "rust_overview.jpg", "type": "image"}],
            "ai_chat": [],
            "other_notes": "",
        },
    ]

    for raw_ticket in sample_tickets_data_raw:
        ticket_id = generate_ticket_id()
        new_ticket = {
            "ticket_id": ticket_id,
            "container_id": raw_ticket["container_id"],
            "company": raw_ticket["company"],
            "total_cost_estimate": raw_ticket["total_cost_estimate"],
            "container_age": raw_ticket["container_age"],
            "repairs": raw_ticket.get("repairs", []),
            "media": raw_ticket.get("media_simulated", raw_ticket.get("media", [])),
            "other_notes": raw_ticket.get("other_notes", ""),
            "submitted_date": get_current_timestamp(),
            "ai_chat": [
                {
                    "timestamp": get_current_timestamp(),
                    "sender": "System",
                    "message": "Ticket submitted.",
                }
            ],
        }

        ai_result = run_ai_pipeline(
            ticket=new_ticket,
            settings=settings,
            use_cohere_aya=True,
        )
        log_msg_action = apply_ai_result(new_ticket, ai_result, source="sample seed")
        st.session_state.tickets.append(new_ticket)
        log_action(ticket_id, f"New Sample Ticket Submitted & {log_msg_action}", ai_result)

    st.sidebar.success("All repair estimates have been processed.")


def ask_ai_about_ticket(ticket: dict, question: str, settings: Settings) -> str | None:
    return chat_with_ai_agent(
        ticket_data=ticket,
        user_question=question,
        chat_history=ticket["ai_chat"],
        repair_codes_requiring_images=st.session_state.repair_codes_needing_images,
        api_key=settings.cohere_api_key,
        model_name=settings.cohere_model,
        error_callback=st.error,
    )


def display_ticket_details(ticket: dict, use_cohere: bool, settings: Settings, expand_details=False):
    with st.expander(
        f"Ticket {ticket['ticket_id']} ({ticket['company']} - {ticket['container_id']}) - Status: {ticket['status']}",
        expanded=expand_details,
    ):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Company:** {ticket['company']}")
            st.write(f"**Container ID:** {ticket['container_id']}")
            st.write(f"**Container Age:** {ticket['container_age']} years")
        with col2:
            st.write(f"**Total Cost Estimate:** ${ticket['total_cost_estimate']:.2f}")
            st.write(f"**Submitted:** {ticket.get('submitted_date', 'N/A')}")
            st.write(f"**Last AI Update:** {ticket.get('ai_processed_date', 'N/A')}")

        st.write("**Suggested Repairs:**")
        if ticket.get("repairs"):
            for repair in ticket["repairs"]:
                st.markdown(
                    f"- `{repair.get('code', 'N/A')}`: {repair.get('description', 'No desc')}"
                )
        else:
            st.write("No repairs listed.")

        st.write("**Media Files:**")
        if ticket.get("media"):
            for media_item in ticket["media"]:
                assoc_code = ""
                if media_item.get("repair_code_association"):
                    assoc_code = f" (for {media_item.get('repair_code_association')})"
                st.markdown(
                    f"- {media_item.get('filename', 'N/A')} ({media_item.get('type', 'N/A')}){assoc_code}"
                )
        else:
            st.write("No media files attached.")

        st.markdown(f"**Other Notes:** {ticket.get('other_notes', 'None')}")

        if ticket.get("ai_reasoning"):
            st.info(
                f"**AI Agent ({ticket.get('ai_agent_type', 'N/A')} on {ticket.get('ai_processed_date', '')}):**\n"
                f"Decision: **{ticket.get('ai_decision', 'N/A')}** "
                f"(Confidence: {ticket.get('ai_confidence', 0.0):.2f})\n"
                f"Reasoning: {ticket.get('ai_reasoning', 'N/A')}"
            )
            if ticket.get("ai_missing_data_request"):
                st.warning(f"AI requests additional data: {ticket['ai_missing_data_request']}")

        if ticket["status"] in [
            "AI Approved",
            "AI Disapproved",
            "Manual Review Required",
            "Additional Data Requested",
        ]:
            st.markdown("---")
            st.write("**Manual Actions:**")
            b_col1, b_col2, b_col3 = st.columns(3)
            override_user = "GSC User"

            if b_col1.button("Manually Approve", key=f"manual_approve_{ticket['ticket_id']}"):
                ticket["status"] = "Manually Approved"
                ticket["manual_override_by"] = override_user
                ticket["manual_override_date"] = get_current_timestamp()
                ticket["ai_chat"].append(
                    {
                        "timestamp": get_current_timestamp(),
                        "sender": override_user,
                        "message": "Ticket Manually Approved.",
                    }
                )
                log_action(ticket["ticket_id"], "Manually Approved", user=override_user)
                st.rerun()

            if b_col2.button(
                "Manually Disapprove", key=f"manual_disapprove_{ticket['ticket_id']}"
            ):
                ticket["status"] = "Manually Disapproved"
                ticket["manual_override_by"] = override_user
                ticket["manual_override_date"] = get_current_timestamp()
                ticket["ai_chat"].append(
                    {
                        "timestamp": get_current_timestamp(),
                        "sender": override_user,
                        "message": "Ticket Manually Disapproved.",
                    }
                )
                log_action(ticket["ticket_id"], "Manually Disapproved", user=override_user)
                st.rerun()

            if ticket["status"] in ["AI Approved", "AI Disapproved"] and b_col3.button(
                "Revert to Manual Review", key=f"revert_{ticket['ticket_id']}"
            ):
                ticket["original_ai_status"] = ticket["status"]
                ticket["status"] = "Manual Review Required"
                ticket["ai_chat"].append(
                    {
                        "timestamp": get_current_timestamp(),
                        "sender": override_user,
                        "message": "AI decision reverted. Moved to Manual Review.",
                    }
                )
                log_action(
                    ticket["ticket_id"],
                    "AI Decision Reverted to Manual Review",
                    user=override_user,
                )
                st.rerun()

        if ticket["status"] == "Additional Data Requested":
            st.markdown("---")
            st.write("**Provide Additional Data:**")
            missing_codes_str = ticket.get("ai_missing_data_request", "")
            codes_to_upload_for = []
            if "missing for repair codes:" in missing_codes_str:
                try:
                    codes_to_upload_for = [
                        code.strip()
                        for code in missing_codes_str.split("missing for repair codes:")[1]
                        .split(".")[0]
                        .split(",")
                    ]
                except Exception:
                    st.caption("Could not automatically parse specific codes from data request.")

            if codes_to_upload_for:
                st.write(
                    f"The AI specifically requested images for codes: {', '.join(codes_to_upload_for)}"
                )

            uploaded_files_for_ticket = st.file_uploader(
                "Upload required files",
                accept_multiple_files=True,
                key=f"file_upload_{ticket['ticket_id']}",
                type=["png", "jpg", "jpeg", "pdf"],
            )

            temp_file_associations = {}
            if uploaded_files_for_ticket:
                st.write("Associate uploaded files with repair codes (if applicable):")
                for index, uploaded_file in enumerate(uploaded_files_for_ticket):
                    relevant_repair_codes = codes_to_upload_for or [
                        repair["code"] for repair in ticket.get("repairs", [])
                    ]
                    assoc_code = st.selectbox(
                        f"Associate '{uploaded_file.name}' with code:",
                        options=[""] + relevant_repair_codes,
                        key=f"assoc_select_{ticket['ticket_id']}_{index}",
                    )
                    temp_file_associations[uploaded_file.name] = {
                        "file": uploaded_file,
                        "association": assoc_code,
                    }

            if st.button(
                "Submit Uploaded Data & Reprocess Ticket",
                key=f"resubmit_data_{ticket['ticket_id']}",
            ):
                if not uploaded_files_for_ticket:
                    st.warning("Please upload files before submitting.")
                else:
                    num_added = 0
                    for data in temp_file_associations.values():
                        uploaded_file = data["file"]
                        association = data["association"]
                        ticket.setdefault("media", []).append(
                            {
                                "filename": uploaded_file.name,
                                "type": uploaded_file.type,
                                "repair_code_association": association if association else None,
                                "uploaded_timestamp": get_current_timestamp(),
                            }
                        )
                        num_added += 1

                    ticket["ai_chat"].append(
                        {
                            "timestamp": get_current_timestamp(),
                            "sender": "System",
                            "message": (
                                f"{num_added} file(s) uploaded by user. "
                                "Resubmitting for AI review."
                            ),
                        }
                    )
                    log_action(
                        ticket["ticket_id"],
                        f"{num_added} File(s) Uploaded for Data Request",
                        user="GSC User",
                    )

                    with st.spinner(
                        f"AI re-processing ticket {ticket['ticket_id']} with new data..."
                    ):
                        ai_result = run_ai_pipeline(
                            ticket=ticket,
                            settings=settings,
                            use_cohere_aya=use_cohere,
                        )

                    log_msg = apply_ai_result(ticket, ai_result, source="resubmission")
                    log_action(ticket["ticket_id"], f"AI Reprocessed: {log_msg}", ai_result)
                    st.rerun()

        if ticket["status"] in ["Manual Review Required", "Additional Data Requested"]:
            st.markdown("---")
            st.write("**Chat with AI Agent & Team Communication:**")
            chat_user = "GSC/Inspector User"

            if not isinstance(ticket.get("ai_chat"), list):
                ticket["ai_chat"] = []

            for chat_msg in ticket["ai_chat"]:
                sender_style = "🤖 " if chat_msg["sender"] == "AI Agent" else ""
                st.markdown(
                    f"<sub>**[{chat_msg['timestamp']}] {sender_style}{chat_msg['sender']}:** "
                    f"{chat_msg['message']}</sub>",
                    unsafe_allow_html=True,
                )

            col1, col2 = st.columns([3, 1])
            with col1:
                new_message = st.text_area(
                    "Your message or question for the AI or Colleagues:",
                    key=f"manual_chat_input_{ticket['ticket_id']}",
                    height=75,
                )
            with col2:
                st.write("Send to:")
                send_to_ai = st.button("Ask AI Agent", key=f"send_ai_chat_{ticket['ticket_id']}")
                send_manual = st.button(
                    "Send to Team", key=f"send_manual_chat_{ticket['ticket_id']}"
                )

            if send_to_ai:
                if new_message:
                    user_msg_data = {
                        "timestamp": get_current_timestamp(),
                        "sender": chat_user,
                        "message": new_message,
                    }
                    ticket["ai_chat"].append(user_msg_data)

                    with st.spinner("AI Agent is thinking..."):
                        ai_response = ask_ai_about_ticket(ticket, new_message, settings)
                        if ai_response:
                            ai_msg_data = {
                                "timestamp": get_current_timestamp(),
                                "sender": "AI Agent",
                                "message": ai_response,
                            }
                            ticket["ai_chat"].append(ai_msg_data)
                            log_action(
                                ticket["ticket_id"],
                                "AI Chat Exchange",
                                {"user_msg": new_message, "ai_response": ai_response},
                                user=chat_user,
                            )
                    st.rerun()
                else:
                    st.warning("Please enter a message or question for the AI.")

            if send_manual:
                if new_message:
                    msg_data = {
                        "timestamp": get_current_timestamp(),
                        "sender": chat_user,
                        "message": new_message,
                    }
                    ticket["ai_chat"].append(msg_data)
                    log_action(
                        ticket["ticket_id"],
                        "Manual Chat Message Added",
                        msg_data,
                        user=chat_user,
                    )
                    st.rerun()
                else:
                    st.warning("Please enter a message.")
        else:
            st.markdown("---")
            st.write("**Communication Log & AI Questions:**")

            if not isinstance(ticket.get("ai_chat"), list):
                ticket["ai_chat"] = []

            if not ticket.get("ai_chat"):
                st.caption("No messages yet.")
            else:
                for chat_msg in ticket.get("ai_chat", []):
                    sender_style = "🤖 " if chat_msg["sender"] == "AI Agent" else ""
                    st.markdown(
                        f"<sub>**[{chat_msg['timestamp']}] {sender_style}{chat_msg['sender']}:** "
                        f"{chat_msg['message']}</sub>",
                        unsafe_allow_html=True,
                    )

            st.write("**Ask the AI Agent about this decision:**")
            ai_question = st.text_input(
                "Question for AI Agent:",
                key=f"ai_question_{ticket['ticket_id']}",
            )
            if st.button("Ask AI", key=f"ask_ai_{ticket['ticket_id']}"):
                if ai_question:
                    chat_user = "User"
                    user_msg_data = {
                        "timestamp": get_current_timestamp(),
                        "sender": chat_user,
                        "message": ai_question,
                    }
                    ticket["ai_chat"].append(user_msg_data)

                    with st.spinner("AI Agent is responding..."):
                        ai_response = ask_ai_about_ticket(ticket, ai_question, settings)
                        if ai_response:
                            ai_msg_data = {
                                "timestamp": get_current_timestamp(),
                                "sender": "AI Agent",
                                "message": ai_response,
                            }
                            ticket["ai_chat"].append(ai_msg_data)
                            log_action(
                                ticket["ticket_id"],
                                "AI Question",
                                {"question": ai_question, "ai_response": ai_response},
                                user=chat_user,
                            )
                    st.rerun()
                else:
                    st.warning("Please enter a question.")


def render_home_page() -> None:
    st.header("Dashboard Overview")

    manual_review_needed = [
        ticket for ticket in st.session_state.tickets if ticket["status"] == "Manual Review Required"
    ]
    additional_data_requested = [
        ticket
        for ticket in st.session_state.tickets
        if ticket["status"] == "Additional Data Requested"
    ]

    ai_approved_count = len(
        [ticket for ticket in st.session_state.tickets if ticket["status"] == "AI Approved"]
    )
    ai_disapproved_count = len(
        [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] == "AI Disapproved"
        ]
    )
    manual_approved_count = len(
        [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] == "Manually Approved"
        ]
    )
    manual_disapproved_count = len(
        [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] == "Manually Disapproved"
        ]
    )

    col1, col2 = st.columns(2)
    col1.metric("Tickets for Manual Review", len(manual_review_needed))
    col2.metric("Tickets Requiring Additional Data", len(additional_data_requested))

    st.subheader("Completion Status")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("AI Approved", ai_approved_count)
    col_b.metric("AI Disapproved", ai_disapproved_count)
    col_c.metric("Manually Approved", manual_approved_count)
    col_d.metric("Manually Disapproved", manual_disapproved_count)

    total_to_do = len(manual_review_needed) + len(additional_data_requested)
    estimated_time_seconds = total_to_do * 180
    st.info(f"**Total active tickets requiring attention:** {total_to_do}")
    if total_to_do > 0:
        st.info(
            f"**Estimated time for manual actions:** {datetime.timedelta(seconds=estimated_time_seconds)}"
        )
    else:
        st.success("All active tickets processed or awaiting external action!")

    st.subheader("Recent Activity Log (Last 10)")
    if st.session_state.logs:
        log_df = pd.DataFrame(st.session_state.logs).sort_values(
            by="timestamp",
            ascending=False,
        )
        st.dataframe(log_df.head(10).astype(str), use_container_width=True)
    else:
        st.write("No activity logged yet.")


def render_training_page(settings: Settings) -> None:
    st.header("AI Evaluation & Configuration")
    st.subheader("AI Decision Thresholds (Cost vs. Age)")

    updated_thresholds = {}
    for age_range, current_threshold in st.session_state.age_cost_thresholds.items():
        updated_thresholds[age_range] = st.number_input(
            f"Max Cost for {age_range}",
            min_value=0,
            value=current_threshold,
            step=50,
            key=f"thresh_{age_range}",
        )

    if st.button("Save Cost Thresholds"):
        st.session_state.age_cost_thresholds = updated_thresholds
        log_action(None, "AI Cost Thresholds Updated", updated_thresholds)
        st.success("Cost thresholds updated!")
        st.rerun()

    st.markdown("---")
    st.subheader("Repair Codes Requiring Images")
    st.markdown("Define which repair codes mandatorily require an image for approval.")
    current_img_req_codes = st.session_state.repair_codes_needing_images

    if current_img_req_codes:
        st.write("Current codes requiring images:")
        for code, desc in current_img_req_codes.items():
            st.markdown(f"- `{code}`: {desc}")
    else:
        st.write("No repair codes currently configured to require images.")

    with st.form("add_image_req_code_form"):
        new_code = st.text_input("New Repair Code (e.g., DMG002)")
        new_code_desc = st.text_input("Description for new code")
        if st.form_submit_button("Add Image Requirement"):
            if new_code and new_code_desc:
                st.session_state.repair_codes_needing_images[new_code.upper()] = new_code_desc
                log_action(None, "AI Image Requirement Added", {new_code.upper(): new_code_desc})
                st.success(f"Added '{new_code.upper()}'.")
                st.rerun()
            else:
                st.error("Code and description required.")

    if current_img_req_codes:
        code_to_remove = st.selectbox(
            "Select code to remove:",
            options=[""] + list(current_img_req_codes.keys()),
            key="remove_code_select",
        )
        if st.button("Remove Selected Code Requirement") and code_to_remove:
            del st.session_state.repair_codes_needing_images[code_to_remove]
            log_action(None, "AI Image Requirement Removed", {"code_removed": code_to_remove})
            st.success(f"Removed '{code_to_remove}'.")
            st.rerun()

    st.subheader("Alternative: Upload SOP Documents")
    st.markdown(
        "Upload Standard Operating Procedure documents to automatically extract "
        "repair codes requiring images."
    )
    st.markdown(
        "For demo purposes, upload the PDF document at "
        f"`{settings.sample_sop_path}`. The system will process it using Mistral OCR "
        "and analyze the content with Cohere to identify relevant repair codes."
    )

    uploaded_sop = st.file_uploader(
        "Upload SOP Document (PDF)",
        type=["pdf"],
        key="sop_uploader",
    )
    if uploaded_sop and st.button("Process SOP Document"):
        with st.spinner("Processing SOP document..."):
            ocr_result = process_sop_with_mistral_ocr(
                uploaded_file=uploaded_sop,
                api_key=settings.mistral_api_key,
                error_callback=st.error,
            )
            if ocr_result:
                st.success("OCR processing completed.")
                with st.spinner("Analyzing document content..."):
                    new_codes = analyze_sop_with_cohere(
                        ocr_result=ocr_result,
                        api_key=settings.cohere_api_key,
                        model_name=settings.cohere_model,
                        error_callback=st.error,
                    )
                    if new_codes:
                        st.session_state.repair_codes_needing_images.update(new_codes)
                        log_action(
                            None,
                            "SOP Analysis Complete",
                            {
                                "new_codes_added": new_codes,
                                "total_codes": len(
                                    st.session_state.repair_codes_needing_images
                                ),
                            },
                        )
                        st.success(
                            f"Added {len(new_codes)} new repair codes requiring images."
                        )
                        st.json(new_codes)
                    else:
                        st.info("No new repair codes requiring images found in the document.")
            else:
                st.error("Failed to process SOP document.")

    st.markdown("---")
    st.subheader("Evaluate AI Decisions")
    ai_processed_tickets = [
        ticket
        for ticket in st.session_state.tickets
        if ticket["status"] in ["AI Approved", "AI Disapproved"] and "ai_decision" in ticket
    ]
    if not ai_processed_tickets:
        st.info("No AI-processed tickets for evaluation yet.")
    else:
        for ticket in ai_processed_tickets:
            with st.container(border=True):
                st.write(
                    f"**Ticket ID: {ticket['ticket_id']}** "
                    f"(AI: {ticket['ai_decision']} @ {ticket['ai_confidence']:.2f})"
                )
                st.caption(f"AI Reasoning: {ticket['ai_reasoning']}")
                existing_feedback = next(
                    (
                        feedback
                        for feedback in st.session_state.feedback
                        if feedback["ticket_id"] == ticket["ticket_id"]
                    ),
                    None,
                )
                if existing_feedback:
                    st.success(
                        f"Feedback: {existing_feedback['evaluation']} "
                        f"({existing_feedback.get('comment', '')})"
                    )
                else:
                    feedback_choice = st.radio(
                        "Your evaluation:",
                        ("Good", "Bad", "Not Set"),
                        index=2,
                        key=f"eval_radio_{ticket['ticket_id']}",
                        horizontal=True,
                    )
                    feedback_comment = st.text_input(
                        "Comment",
                        key=f"comment_fb_{ticket['ticket_id']}",
                    )
                    if st.button("Submit Feedback", key=f"submit_fb_{ticket['ticket_id']}"):
                        if feedback_choice != "Not Set":
                            feedback_data = {
                                "ticket_id": ticket["ticket_id"],
                                "evaluation": feedback_choice,
                                "comment": feedback_comment,
                                "timestamp": get_current_timestamp(),
                            }
                            st.session_state.feedback.append(feedback_data)
                            log_action(
                                ticket["ticket_id"],
                                "AI Feedback Submitted",
                                feedback_data,
                            )
                            st.success("Feedback submitted.")
                            st.rerun()
                        else:
                            st.warning("Please select 'Good' or 'Bad' for evaluation.")

    if st.session_state.feedback:
        st.subheader("Collected Feedback Data")
        feedback_df = pd.DataFrame(st.session_state.feedback)
        st.dataframe(feedback_df.astype(str), use_container_width=True)
        st.download_button(
            "Download Feedback (CSV)",
            feedback_df.to_csv(index=False).encode("utf-8"),
            "ai_feedback.csv",
            "text/csv",
        )


def render_approvals_page(use_cohere: bool, settings: Settings) -> None:
    st.header("Repair Ticket Approval Queues")
    ticket_categories_map = {
        "Manual Review Required": [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] == "Manual Review Required"
        ],
        "Additional Data Requested": [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] == "Additional Data Requested"
        ],
        "AI Approved": [
            ticket for ticket in st.session_state.tickets if ticket["status"] == "AI Approved"
        ],
        "AI Disapproved": [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] == "AI Disapproved"
        ],
        "Manually Processed": [
            ticket
            for ticket in st.session_state.tickets
            if ticket["status"] in ["Manually Approved", "Manually Disapproved"]
        ],
    }
    tab_titles_with_counts = [
        f"{title} ({len(tickets)})" for title, tickets in ticket_categories_map.items()
    ]
    tabs = st.tabs(tab_titles_with_counts)
    for index, (title, tickets_in_category) in enumerate(ticket_categories_map.items()):
        with tabs[index]:
            if not tickets_in_category:
                st.info(f"No tickets in '{title}' queue.")
            else:
                for ticket in sorted(
                    tickets_in_category,
                    key=lambda item: item.get("submitted_date", ""),
                    reverse=True,
                ):
                    display_ticket_details(ticket, use_cohere, settings)
                    st.markdown("---")


def render_submit_page(use_cohere: bool, settings: Settings) -> None:
    st.header("Submit New Repair Ticket")
    with st.form("new_ticket_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        container_id = c1.text_input("Container ID*", key="submit_cid")
        company = c2.selectbox(
            "Shipping Company*",
            ["Maersk", "Cosco", "CGM", "MSC", "Hapag", "Other"],
            key="submit_comp",
        )
        if company == "Other":
            company = c2.text_input("Specify Other Company", key="submit_comp_other")
        total_cost_estimate = c1.number_input(
            "Total Cost Estimate ($)*",
            min_value=0.0,
            step=10.0,
            format="%.2f",
            key="submit_cost",
        )
        container_age = c2.number_input(
            "Container Age (years)*",
            min_value=0,
            max_value=50,
            step=1,
            key="submit_age",
        )

        st.markdown("**Suggested Repairs:**")
        rc1, rc2, rc3 = st.columns([2, 3, 1])
        repair_code_input = rc1.text_input("Code", key="new_repair_code")
        repair_desc_input = rc2.text_input("Description", key="new_repair_desc")
        if rc3.form_submit_button("Add Repair"):
            if repair_code_input and repair_desc_input:
                st.session_state.current_repairs_for_new_ticket.append(
                    {
                        "code": repair_code_input.upper(),
                        "description": repair_desc_input,
                    }
                )
                st.success(f"Added repair: {repair_code_input.upper()}")
            else:
                st.warning("Repair code and description needed.")

        if st.session_state.current_repairs_for_new_ticket:
            st.write("Current repairs for this ticket:")
            for repair in st.session_state.current_repairs_for_new_ticket:
                st.markdown(f"- `{repair['code']}`: {repair['description']}")

        other_notes = st.text_area("Other Notes / Observations", key="submit_notes")

        st.markdown("**Attach Media Files:**")
        uploaded_files = st.file_uploader(
            "Upload images, videos, or documents",
            accept_multiple_files=True,
            key="submit_media_uploader",
            type=["png", "jpg", "jpeg", "mp4", "pdf", "mov"],
        )
        st.session_state.current_media_for_new_ticket_with_assoc = []
        if uploaded_files:
            st.write("Associate uploaded files with repair codes (optional):")
            current_repair_codes_in_ticket = [
                repair["code"] for repair in st.session_state.current_repairs_for_new_ticket
            ]
            for index, uploaded_file in enumerate(uploaded_files):
                assoc_code = st.selectbox(
                    f"Associate '{uploaded_file.name}' with repair code:",
                    options=[""] + current_repair_codes_in_ticket,
                    key=f"submit_media_assoc_{index}",
                )
                st.session_state.current_media_for_new_ticket_with_assoc.append(
                    {
                        "original_file_obj": uploaded_file,
                        "filename": uploaded_file.name,
                        "type": uploaded_file.type,
                        "repair_code_association": assoc_code if assoc_code else None,
                    }
                )

        submitted_ticket_button = st.form_submit_button(
            "Submit New Ticket for AI Processing"
        )

        if submitted_ticket_button:
            if not container_id or not company or total_cost_estimate <= 0:
                st.error("Please fill required fields: Container ID, Company, Cost Estimate.")
            else:
                ticket_id = generate_ticket_id()
                final_media_list = [
                    {
                        "filename": media_item_info["filename"],
                        "type": media_item_info["type"],
                        "repair_code_association": media_item_info["repair_code_association"],
                    }
                    for media_item_info in st.session_state.current_media_for_new_ticket_with_assoc
                ]

                new_ticket = {
                    "ticket_id": ticket_id,
                    "container_id": container_id,
                    "company": company,
                    "total_cost_estimate": total_cost_estimate,
                    "container_age": container_age,
                    "repairs": list(st.session_state.current_repairs_for_new_ticket),
                    "media": final_media_list,
                    "other_notes": other_notes,
                    "submitted_date": get_current_timestamp(),
                    "ai_chat": [
                        {
                            "timestamp": get_current_timestamp(),
                            "sender": "System",
                            "message": "Ticket submitted for AI processing.",
                        }
                    ],
                }

                with st.spinner(f"AI processing new ticket {ticket_id}..."):
                    ai_result = run_ai_pipeline(
                        ticket=new_ticket,
                        settings=settings,
                        use_cohere_aya=use_cohere,
                    )

                log_msg_action = apply_ai_result(new_ticket, ai_result, source="new ticket")
                st.session_state.tickets.append(new_ticket)
                log_action(ticket_id, f"New Ticket Submitted & {log_msg_action}", ai_result)
                st.session_state.current_repairs_for_new_ticket = []
                st.session_state.current_media_for_new_ticket_with_assoc = []
                st.success(
                    f"New ticket {ticket_id} submitted and processed. Status: {new_ticket['status']}"
                )


def main() -> None:
    settings = get_settings()
    st.set_page_config(layout="wide", page_title="AI Repair Ticket Approval")
    init_session_state()
    add_sample_tickets_if_needed(settings)

    st.sidebar.title("Repair Approval Platform")
    st.sidebar.markdown("---")
    page_options = [
        "Home (GSC)",
        "AI Training & Settings (GSC)",
        "Approvals (GSC & Inspectors)",
        "Submit New Estimate (Inspectors)",
    ]
    page = st.sidebar.radio("Navigation", page_options)
    st.sidebar.markdown("---")
    st.sidebar.subheader("AI Agent Settings")
    use_cohere = st.sidebar.checkbox("Use Cohere Aya Model", value=True)
    if not use_cohere:
        st.sidebar.caption("Using Custom Model Placeholder.")

    if page == "Home (GSC)":
        render_home_page()
    elif page == "AI Training & Settings (GSC)":
        render_training_page(settings)
    elif page == "Approvals (GSC & Inspectors)":
        render_approvals_page(use_cohere, settings)
    elif page == "Submit New Estimate (Inspectors)":
        render_submit_page(use_cohere, settings)

    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 AI Repair Co. (Demo v4)")
    st.sidebar.markdown("### AI Agent Strategy Note")
    st.sidebar.caption(
        "Hybrid Approach (Recommended): Rule-Based Engine for deterministic checks "
        "(cost, mandatory images), LLM for complex cases/text, and specialized vision "
        "models for image analysis. This demo uses LLM reasoning with rule-based checks."
    )


if __name__ == "__main__":
    main()
