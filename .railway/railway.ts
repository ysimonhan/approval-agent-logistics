import { defineRailway, github, project, service } from "railway/iac";

export const partial = "approval-agent-logistics";

export default defineRailway(() => {
  const app = service("approval-agent-logistics", {
    source: github("ysimonhan/approval-agent-logistics", {
      branch: "main",
      checkSuites: false,
    }),
    build: {
      buildEnvironment: "V3",
      builder: "DOCKERFILE",
      dockerfilePath: "infrastructure/Dockerfile",
    },
    start:
      '/bin/sh -c "exec python -m streamlit run src/logistics_approval_agent/web/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"',
    healthcheck: "/",
    healthcheckTimeout: 300,
    replicas: { "europe-west4-drams3a": 1 },
    deploy: {
      // Railway normalizes default restartPolicyType ("ON_FAILURE") to null;
      // setting restartPolicyMaxRetries: 3 preserves ON_FAILURE x3 with zero plan drift.
      restartPolicyMaxRetries: 3,
      sleepApplication: true,
      limitOverride: {
        containers: {
          cpu: 4,
          memoryBytes: 4000000000,
        },
      },
    },
  });

  return project("approval-agent-logistics", {
    variables: { managed: false },
    resources: [app],
  });
});
