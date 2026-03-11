# Yahoo Fantasy API: Roster & Transaction Reference (XML)

## 1. Overview
Yahoo Fantasy API uses OAuth2 for authentication and XML for data exchange. While Python libraries like `yahoo-fantasy-api` are available, this reference covers the raw XML structures for `set_lineup` and `add/drop` actions.

## 2. Setting the Daily Lineup (PUT)
### **Endpoint**
`PUT https://fantasysports.yahooapis.com/fantasy/v2/team/{team_key}/roster`

### **XML Structure**
```xml
<fantasy_content>
  <roster>
    <coverage_type>date</coverage_type>
    <date>2026-03-11</date>
    <players>
      <player>
        <player_key>441.p.9557</player_key> <!-- Example Player: Aaron Judge -->
        <selected_position>OF</selected_position>
      </player>
      <player>
        <player_key>441.p.10001</player_key>
        <selected_position>BN</selected_position> <!-- Bench -->
      </player>
    </players>
  </roster>
</fantasy_content>
```

## 3. Adding and Dropping Players (POST)
### **Endpoint**
`POST https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/transactions`

### **XML Structure (Add and Drop)**
```xml
<fantasy_content>
  <transaction>
    <type>add/drop</type>
    <players>
      <!-- Add Player -->
      <player>
        <player_key>441.p.12345</player_key>
        <transaction_data>
          <type>add</type>
          <destination_team_key>{team_key}</destination_team_key>
        </transaction_data>
      </player>
      <!-- Drop Player -->
      <player>
        <player_key>441.p.99999</player_key>
        <transaction_data>
          <type>drop</type>
          <source_team_key>{team_key}</source_team_key>
        </transaction_data>
      </player>
    </players>
  </transaction>
</fantasy_content>
```

## 4. Error Handling
*   **Error 25:** "Player is already on a team." Occurs if `destination_team_key` is incorrect or missing.
*   **Error 101:** "Roster is full." Ensure you are submitting an `add/drop` paired transaction to keep roster spots balanced.
*   **Error 217:** "Invalid position." Ensure the `selected_position` matches the player's eligible positions in the league settings.
