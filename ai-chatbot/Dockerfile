FROM node:23-alpine
RUN corepack enable

RUN apk add --update python3 make g++

COPY . /app
WORKDIR /app
RUN rm -rf node_modules


RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile

RUN pnpm run build
