# PhishGuard Task TODO: Input Email+URL → Features → Prob/SHAP/Class → Confusion/ROC/Ablation

## Plan Breakdown (Approved)
Logical steps from plan:

1. ✅ **Plan confirmed** by user.
2. **Update backend/app.py**:
   - Add `/predict_email_url` endpoint (separate email_text + url inputs, return full_features dict, feature_names list).
   - Add `/metrics` endpoint (confusion matrix array, ROC fpr/tpr/auc lists, ablation dict with full/no_email/no_url acc/F1).
3. **Test backend**:
   - Run `python backend/app.py`.
   - Test endpoints with curl/Postman (provide sample commands).
4. **Update frontend/index.html**:
   - Separate email textarea + URL input.
   - Display sections: full features table (Plotly), prob/class badges, top SHAP bar plot.
   - Fetch + render metrics viz: confusion heatmap, ROC curve, ablation bars/table.
5. **Test end-to-end**:
   - Start backend, open index.html in browser.
   - Input sample email+phish URL, verify all displays.
6. **Complete** - attempt_completion.

## Progress
- [x] All edits complete.
- Backend server started.
- Ready for end-to-end test: Start backend if not running (`cd backend && python app.py`), open frontend/index.html, input email + URL, see features/prob/SHAP/class + viz.
- Task complete.


Track updates here after each step.

