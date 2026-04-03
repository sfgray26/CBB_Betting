/**
 * Core Decision Contracts
 * 
 * These contracts are the immutable records that flow through the system.
 * They are the single source of truth for all decisions.
 * 
 * Rules:
 * - All properties are readonly
 * - No methods, only data
 * - Serializable to JSON
 * - Versioned for migration safety
 */

// ============================================================================
// Shared Primitives
// ============================================================================

export type UUID = string;
export type ISO8601Timestamp = string;
export type Percentage = number; // 0.0 - 1.0

export interface PlayerIdentity {
  readonly id: UUID;
  readonly mlbamId: string;           // MLB Stats API ID
  readonly name: string;
  readonly team: string;
  readonly position: string[];
}

export interface FantasyEntity {
  readonly type: 'player' | 'team' | 'lineup_slot';
  readonly id: UUID;
  readonly platformId?: string;       // Platform-specific ID
}

// ============================================================================
// Decision Contract 1: LineupOptimizationRequest
// ============================================================================

export interface LineupOptimizationRequest {
  readonly id: UUID;
  readonly version: 'v1';
  readonly createdAt: ISO8601Timestamp;
  
  // Context
  readonly leagueConfig: LeagueConfiguration;
  readonly scoringPeriod: ScoringPeriod;
  readonly rosterConstraints: RosterConstraints;
  
  // Inputs
  readonly availablePlayers: PlayerPool;
  readonly optimizationObjective: OptimizationObjective;
  readonly riskTolerance: RiskProfile;
  
  // Optional constraints
  readonly weatherSensitivity?: WeatherSensitivity;
  readonly correlationPreferences?: CorrelationPreferences;
  readonly manualOverrides?: ManualOverride[];
}

export interface LeagueConfiguration {
  readonly platform: 'yahoo' | 'espn' | 'fantrax' | 'sleeper' | 'custom';
  readonly format: 'h2h' | 'roto' | 'points';
  readonly scoringRules: ScoringRules;
  readonly rosterPositions: RosterPosition[];
  readonly leagueSize: number;
  readonly categories?: string[];     // For roto leagues
}

export interface ScoringRules {
  readonly batting: Record<string, number>;
  readonly pitching: Record<string, number>;
}

export interface RosterPosition {
  readonly slot: string;              // "C", "1B", "2B", "UTIL", "SP", etc.
  readonly maxCount: number;
  readonly eligiblePositions: string[];
}

export interface ScoringPeriod {
  readonly type: 'daily' | 'weekly' | 'season';
  readonly startDate: ISO8601Timestamp;
  readonly endDate: ISO8601Timestamp;
  readonly games: ScheduledGame[];
}

export interface ScheduledGame {
  readonly id: string;
  readonly homeTeam: string;
  readonly awayTeam: string;
  readonly startTime: ISO8601Timestamp;
  readonly ballpark: string;
  readonly weatherForecast?: WeatherForecast;
}

export interface PlayerPool {
  readonly players: PoolPlayer[];
  readonly lastUpdated: ISO8601Timestamp;
}

export interface PoolPlayer {
  readonly player: PlayerIdentity;
  readonly isAvailable: boolean;
  readonly currentRosterStatus?: 'starting' | 'bench' | 'injured' | 'minors';
  readonly acquisitionCost?: number;  // For auction/salary cap leagues
}

export interface RosterConstraints {
  readonly lockedSlots: string[];     // Positions already decided
  readonly mustInclude?: UUID[];      // Force-include these players
  readonly mustExclude?: UUID[];      // Force-exclude these players
  readonly maxExposurePerTeam?: Percentage;
}

export interface OptimizationObjective {
  readonly type: 'maximize_expected' | 'maximize_floor' | 'maximize_ceiling' | 'balanced';
  readonly constraints?: {
    readonly maxExposurePerPlayer?: Percentage;
    readonly stackPreferences?: StackPreference[];
    readonly diversificationTarget?: Percentage;
  };
}

export interface StackPreference {
  readonly team: string;
  readonly weight: number;            // 0.0 - 1.0, preference strength
  readonly maxPlayers: number;
}

export type RiskProfile = 
  | { readonly type: 'conservative'; varianceTolerance: 0.1; description: 'Minimize downside' }
  | { readonly type: 'balanced'; varianceTolerance: 0.3; description: 'Balance risk and reward' }
  | { readonly type: 'aggressive'; varianceTolerance: 0.5; description: 'Maximize upside potential' };

export interface WeatherSensitivity {
  readonly rainThreshold: Percentage; // Cancel play if rain prob > this
  readonly windThreshold: number;     // mph, factor for fly ball hitters
  readonly temperatureThreshold: { min: number; max: number };
}

export interface CorrelationPreferences {
  readonly favorTeamStacks: boolean;
  readonly avoidPitcherVsBatter: boolean;
  readonly favorOpposingLineups: boolean; // For game stack in DFS
}

export interface ManualOverride {
  readonly playerId: UUID;
  readonly action: 'lock_in' | 'lock_out' | 'boost_projection';
  readonly value?: number;            // For boost_projection
  readonly reason?: string;
}

// ============================================================================
// Decision Contract 2: PlayerValuationReport
// ============================================================================

export interface PlayerValuationReport {
  readonly id: UUID;
  readonly version: 'v1';
  readonly generatedAt: ISO8601Timestamp;
  readonly validUntil: ISO8601Timestamp;
  
  readonly player: PlayerIdentity;
  readonly context: ValuationContext;
  
  // Core valuations
  readonly pointProjection: Distribution;
  readonly valueOverReplacement: number;
  readonly positionalScarcity: PositionalScarcity;
  
  // Risk-adjusted metrics
  readonly riskProfile: PlayerRiskProfile;
  readonly floorProjection: number;
  readonly ceilingProjection: number;
  
  // Contextual factors
  readonly factors: AppliedFactor[];
  
  // Audit trail
  readonly methodology: ValuationMethodology;
  readonly dataSources: DataSourceReference[];
}

export interface Distribution {
  readonly mean: number;
  readonly median: number;
  readonly standardDeviation: number;
  readonly variance: number;
  readonly skewness: number;
  readonly kurtosis: number;
  readonly percentiles: PercentileDistribution;
  readonly histogramBins?: HistogramBin[];
  readonly samples?: number[];        // Raw samples if distribution is empirical
}

export interface PercentileDistribution {
  readonly p5: number;
  readonly p10: number;
  readonly p25: number;
  readonly p50: number;
  readonly p75: number;
  readonly p90: number;
  readonly p95: number;
}

export interface HistogramBin {
  readonly min: number;
  readonly max: number;
  readonly count: number;
  readonly density: Percentage;
}

export interface ValuationContext {
  readonly scoringPeriod: ScoringPeriod;
  readonly leagueScoring: ScoringRules;
  readonly opponent?: string;         // For pitcher matchups
  readonly ballpark?: string;
  readonly lineupSpot?: number;       // 1-9
}

export interface PositionalScarcity {
  readonly position: string;
  readonly replacementLevel: number;
  readonly availableAlternatives: number;
  readonly scarcityScore: number;     // Higher = more scarce
}

export interface PlayerRiskProfile {
  readonly injuryRisk: RiskLevel;
  readonly playingTimeRisk: RiskLevel;
  readonly performanceVariance: RiskLevel;
  readonly overallRisk: RiskLevel;
  readonly confidenceInterval: { lower: number; upper: number };
}

export type RiskLevel = 'low' | 'moderate' | 'high' | 'extreme';
export type ConfidenceLevel = 'very_low' | 'low' | 'moderate' | 'high' | 'very_high';

export interface AppliedFactor {
  readonly factorType: FactorType;
  readonly impact: number;            // Multiplier effect (1.0 = neutral)
  readonly confidence: ConfidenceLevel;
  readonly rawData: FactorRawData;
}

export type FactorType = 
  | 'weather'
  | 'ballpark'
  | 'platoon_split'
  | 'rest'
  | 'momentum'
  | 'lineup_position'
  | 'opponent_quality'
  | 'umpire';

export type FactorRawData =
  | WeatherFactor
  | BallparkFactor
  | PlatoonSplitFactor
  | RestFactor
  | MomentumFactor
  | LineupPositionFactor
  | OpponentQualityFactor;

export interface WeatherFactor {
  readonly temperature: number;
  readonly humidity: Percentage;
  readonly windSpeed: number;
  readonly windDirection: string;
  readonly precipitationChance: Percentage;
  readonly gameTime: ISO8601Timestamp;
}

export interface BallparkFactor {
  readonly parkName: string;
  readonly runsFactor: number;
  readonly hrFactor: number;
  readonly avgFactor: number;
  readonly obpFactor: number;
  readonly slgFactor: number;
  readonly handednessSplit?: { vsLeft: number; vsRight: number };
}

export interface PlatoonSplitFactor {
  readonly vsHandedness: 'L' | 'R' | 'S';
  readonly careerSplit: number;       // wOBA vs this handedness
  readonly seasonSplit: number;
  readonly last30DaysSplit: number;
}

export interface RestFactor {
  readonly daysRest: number;
  readonly isDayGameAfterNightGame: boolean;
  readonly travelDistance: number;    // Miles traveled
}

export interface MomentumFactor {
  readonly last7DaysWoba: number;
  readonly last14DaysWoba: number;
  readonly hardHitRateTrend: 'up' | 'stable' | 'down';
  readonly barrelsPerPA: number;
}

export interface LineupPositionFactor {
  readonly battingOrder: number;
  readonly expectedPlateAppearances: number;
  readonly runsRBIContext: 'good' | 'neutral' | 'poor';
}

export interface OpponentQualityFactor {
  readonly opponentTeam: string;
  readonly pitcherName: string;
  readonly pitcherHandedness: 'L' | 'R';
  readonly pitcherERA: number;
  readonly pitcherWHIP: number;
  readonly opponentBullpenStrength: number;
}

export interface ValuationMethodology {
  readonly modelType: 'monte_carlo' | 'ensemble' | 'regression' | 'hybrid';
  readonly simulationCount?: number;
  readonly featuresUsed: string[];
  readonly modelVersion: string;
  readonly trainedAt: ISO8601Timestamp;
}

export interface DataSourceReference {
  readonly source: string;
  readonly endpoint: string;
  readonly fetchedAt: ISO8601Timestamp;
  readonly cacheKey: string;
}

// ============================================================================
// Decision Contract 3: ExecutionDecision
// ============================================================================

export interface ExecutionDecision {
  readonly id: UUID;
  readonly version: 'v1';
  readonly createdAt: ISO8601Timestamp;
  
  readonly decisionType: DecisionType;
  readonly target: FantasyEntity;
  readonly recommendedAction: RecommendedAction;
  
  // Decision rationale
  readonly reasoning: DecisionReasoning;
  readonly confidence: ConfidenceLevel;
  readonly alternativeActions: AlternativeAction[];
  
  // Safety controls
  readonly executionMode: ExecutionMode;
  readonly humanReviewRequired: boolean;
  readonly autoExecuteConditions?: AutoExecuteCondition[];
  
  // Audit
  readonly sourceRequestId?: UUID;    // Links back to originating request
  readonly traceId: UUID;             // For distributed tracing
}

export type DecisionType =
  | 'set_lineup'
  | 'add_player'
  | 'drop_player'
  | 'claim_waiver'
  | 'trade_proposal'
  | 'trade_accept'
  | 'trade_reject'
  | 'sit_player'
  | 'start_player'
  | 'move_to_dl'
  | 'activate_from_dl';

export interface RecommendedAction {
  readonly type: string;
  readonly parameters: Record<string, unknown>;
  readonly expectedOutcome: ExpectedOutcome;
  readonly riskAssessment: RiskAssessment;
  readonly urgency: 'low' | 'medium' | 'high' | 'critical';
}

export interface ExpectedOutcome {
  readonly pointImpact: Distribution;
  readonly categoryImpacts?: Record<string, number>; // For roto
  readonly winProbabilityDelta?: Percentage;
  readonly description: string;
}

export interface RiskAssessment {
  readonly downsideScenario: string;
  readonly worstCaseImpact: number;
  readonly probabilityOfSuccess: Percentage;
  readonly keyRisks: string[];
}

export interface DecisionReasoning {
  readonly summary: string;
  readonly primaryFactor: string;
  readonly supportingFactors: string[];
  readonly counterIndicators: string[];
  readonly keyAssumptions: string[];
  readonly projectionDelta: number;
  readonly comparisonPlayers?: PlayerIdentity[];
}

export interface AlternativeAction {
  readonly action: string;
  readonly parameters: Record<string, unknown>;
  readonly expectedValue: number;
  readonly confidence: ConfidenceLevel;
  readonly whyNotRecommended: string;
}

export type ExecutionMode =
  | { readonly type: 'manual_review'; reason: string }
  | { readonly type: 'suggest_only' }
  | { 
      readonly type: 'auto_if_confident'; 
      minConfidence: ConfidenceLevel; 
      maxRiskScore: number;
      requiredConditions: string[];
    }
  | { 
      readonly type: 'full_auto'; 
      constraints: AutoExecuteConstraint[];
      dailyDigest: boolean;
    };

export interface AutoExecuteCondition {
  readonly metric: string;
  readonly operator: 'gt' | 'lt' | 'eq' | 'between' | 'in';
  readonly value: number | [number, number] | string[];
  readonly currentValue: number | string;
  readonly satisfied: boolean;
}

export interface AutoExecuteConstraint {
  readonly type: 'max_daily_moves' | 'preserve_faab' | 'avoid_drop_list' | 'position_depth';
  readonly parameters: Record<string, unknown>;
}

// ============================================================================
// Result Contracts (Outputs of the system)
// ============================================================================

export interface LineupOptimizationResult {
  readonly requestId: UUID;
  readonly generatedAt: ISO8601Timestamp;
  readonly optimalLineup: LineupSlot[];
  readonly expectedPoints: Distribution;
  readonly confidenceScore: number;
  readonly alternativeLineups: AlternativeLineup[];
  readonly explanation: LineupExplanation;
}

export interface LineupSlot {
  readonly position: string;
  readonly player: PlayerIdentity;
  readonly projectedPoints: number;
  readonly confidence: ConfidenceLevel;
  readonly factors: string[];
}

export interface AlternativeLineup {
  readonly lineup: LineupSlot[];
  readonly expectedPoints: number;
  readonly varianceVsOptimal: number;
  readonly tradeoffDescription: string;
}

export interface LineupExplanation {
  readonly summary: string;
  readonly keyDecisions: KeyDecisionPoint[];
  readonly riskFactors: string[];
  readonly opportunities: string[];
}

export interface KeyDecisionPoint {
  readonly position: string;
  readonly chosenPlayer: PlayerIdentity;
  readonly alternativesConsidered: PlayerIdentity[];
  readonly whyChosen: string;
}

// ============================================================================
// Event Contracts (For async communication)
// ============================================================================

export interface DecisionEvent {
  readonly eventId: UUID;
  readonly eventType: 
    | 'optimization_requested'
    | 'valuation_completed'
    | 'decision_created'
    | 'decision_approved'
    | 'decision_rejected'
    | 'decision_executed'
    | 'execution_failed';
  readonly timestamp: ISO8601Timestamp;
  readonly payload: unknown;
  readonly metadata: EventMetadata;
}

export interface EventMetadata {
  readonly source: string;
  readonly traceId: UUID;
  readonly correlationId?: UUID;
  readonly userId?: string;
}

// ============================================================================
// Configuration Contracts
// ============================================================================

export interface SystemConfiguration {
  readonly version: 'v1';
  readonly automationLevel: 0 | 1 | 2 | 3;
  readonly dataRefreshIntervals: DataRefreshIntervals;
  readonly notificationPreferences: NotificationPreferences;
  readonly riskThresholds: RiskThresholds;
}

export interface DataRefreshIntervals {
  readonly liveScores: number;        // seconds
  readonly playerData: number;        // seconds
  readonly weatherData: number;       // seconds
  readonly projections: number;       // seconds
}

export interface NotificationPreferences {
  readonly email?: string;
  readonly webhookUrl?: string;
  readonly notifyOn: NotificationTrigger[];
}

export type NotificationTrigger =
  | 'high_confidence_decision'
  | 'manual_review_required'
  | 'execution_completed'
  | 'execution_failed'
  | 'daily_digest';

export interface RiskThresholds {
  readonly maxExposurePerPlayer: Percentage;
  readonly maxExposurePerTeam: Percentage;
  readonly minConfidenceForAuto: ConfidenceLevel;
  readonly maxRiskForAuto: RiskLevel;
}
