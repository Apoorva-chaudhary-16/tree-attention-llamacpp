# Orbit API Reference

The Orbit API lets you manage satellites and ground stations over HTTPS. All responses are JSON.

## Authentication

Every request needs a bearer token in the `Authorization` header.

### Getting a token
Call `POST /v2/token` with your client id and secret. The token is valid for 45 minutes.

### Rate limits
Each token may send 120 requests per minute. Extra requests receive status 429.

## Endpoints

### List satellites
```http
GET /v2/satellites?limit=50
```
Returns up to 50 satellites per page. The maximum limit is 200.

### Create a ground station
```http
POST /v2/stations
Content-Type: application/json

{"name": "Bengaluru-1", "lat": 12.97, "lon": 77.59}
```
The call returns status 201 and the new station id.

### Delete a station
```http
DELETE /v2/stations/{id}
```
Deleting a station cannot be undone. The call returns status 204.

## Errors

| Status | Meaning |
|---|---|
| 400 | Malformed request body |
| 401 | Missing or expired token |
| 404 | Unknown satellite or station |
| 429 | Rate limit exceeded |

## Changelog

- 2.1: added the `limit` query parameter
- 2.0: switched from API keys to bearer tokens
- 1.4: first public release
