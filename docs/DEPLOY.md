# Deploying the live app

The `Dockerfile` builds the web app and the Python package into one image that serves both. It starts
`miraichem serve --host 0.0.0.0 --allow-remote --no-allow-hardware`, so a deployed copy can never reach IBM.
No IBM token is copied into the image (`.env` is in `.dockerignore`), and it should not be set on the host.

The image contains the published results only. Runs started on the host are saved to its temporary disk and
are lost when it restarts.

The app has no login, so anyone with the link can start simulator runs and sweeps. Share the link only with
people you trust, or use the offline demo for a public link.

## Run it locally

    docker build -t miraichem .
    docker run --rm -p 8000:8000 miraichem

Then open http://localhost:8000.

## Render (free plan)

1. Sign in at render.com and choose New > Blueprint.
2. Connect the GitHub repository `S-tevens/miraichem`; Render reads `render.yaml`.
3. Deploy. The first build takes several minutes. A free service sleeps when idle and takes about a minute
   to wake.
